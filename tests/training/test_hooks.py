import pytest

import torch

from omegaconf import open_dict
from hydra.utils import get_class

from tests.conftest import distributed_test, config


def _test_hooks_dist(cfg):
    import time
    import torch
    import pandas as pd
    from pathlib import Path
    from ray.train import report
    from utils.context import process_rank, barrier
    from training.loggers import LocalEventWriter
    from training.hooks import (
        AnomalyDetector,
        SamplerSetter,
        LRScheduler,
        IterationTimer,
        PeriodicWriter,
        TorchMemoryStats,
        BestMetricSaver,
        TorchProfiler,
        EarlyStopHook,
        EMASchedulerHook
    )

    success = True

    trainer_cls = get_class(cfg.trainer)
    trainer = trainer_cls(cfg)

    for hook in trainer._hooks:
        if isinstance(hook, AnomalyDetector):
            anomaly_detector_hook = hook
        elif isinstance(hook, SamplerSetter):
            sampler_setter = hook
        elif isinstance(hook, LRScheduler):
            lr_hook = hook
        elif isinstance(hook, IterationTimer):
            timer = hook
        elif isinstance(hook, PeriodicWriter):
            periodic = hook
        elif isinstance(hook, TorchMemoryStats):
            mem_hook = hook
        elif isinstance(hook, BestMetricSaver):
            saver = hook
        elif isinstance(hook, TorchProfiler):
            prof_hook = hook
        elif isinstance(hook, EarlyStopHook):
            ehook = hook
        elif isinstance(hook, EMASchedulerHook):
            ema_hook = hook

    # ---- ---- ---- anomaly detector hook tests ---- ---- ----

    # enter anomaly-detect context for the epoch
    anomaly_detector_hook.before_epoch()

    # feed 5 NaNs, should just increment loss_nans
    for i in range(5):
        anomaly_detector_hook.after_step(data_sample=None, 
                                         outputs=None, 
                                         loss_dict={"step_loss": torch.tensor(float("nan"))})
        assert anomaly_detector_hook.loss_nans == i + 1

    # on the 6th NaN should get an Exception
    success = False
    try:
        anomaly_detector_hook.after_step(data_sample=None, 
                                         outputs=None, 
                                         loss_dict={"step_loss": torch.tensor(float("nan"))})
    except Exception:
        success = True

    if not success:
        raise ValueError("AnomalyDetector did not raise \
                         an exception on the 6th NaN")

    # exit the anomaly context cleanly
    anomaly_detector_hook.after_epoch()
    
    # ---- ---- ---- SamplerSetter tests ---- ---- ----

    if torch.cuda.device_count() > 1:
        # pick a fake epoch to test
        trainer._epoch = 5

        sampler_setter.before_epoch()

        # now the DataLoader's sampler should have recorded that same epoch
        assert trainer.train_dataloader.sampler.epoch == 5, (
            f"Expected sampler.epoch==5 but got "
            f"{trainer.train_dataloader.sampler.epoch}"
        )
    
    # ---- ---- ---- LRScheduler tests ---- ---- ----

    # initialize learning rate scheduler
    lr_hook.before_train()
    best_id = lr_hook._best_param_group_id
    initial_lr = trainer.opt.param_groups[best_id]["lr"]

    # simulate two steps at epoch 0
    trainer._epoch = 0
    trainer._iter = 0
    lr_hook.after_step(data_sample=None, outputs=None, loss_dict={})
    trainer._iter = 1
    lr_hook.after_step(data_sample=None, outputs=None, loss_dict={})

    # pull out what got recorded
    recorded = trainer.event_recorder.get_step_scalars().get("lr", [])
    lrs = [val for val, it, ep in recorded]
    epochs = [ep for val, it, ep in recorded]

    # success only if exactly two entries AND both equal the initial LR
    lrs = (len(lrs) == 2 and all(lr == initial_lr for lr in lrs))
    epochs = (len(epochs) == 2 and all(ep == 0 for ep in epochs))
    assert lrs and epochs, \
        f"Expected 2 recorded LRs at epoch 0, got {len(lrs)} with values: {lrs} " \
        f"and epochs: {epochs}"

    # ---- ---- ---- IterationTimer tests ---- ---- ----
    
    # ensure predictable counters
    trainer.start_iter, trainer._iter, trainer._epoch = 0, 0, 0
    trainer._max_epochs = 1

    warmup = timer._warmup_iter
    rec = trainer.event_recorder

    # ------------------------------------------------------------------ #
    # TRAIN loop simulation (5 steps)
    # ------------------------------------------------------------------ #

    timer.before_train()
    for step in range(5):
        # resets step timer and resumes total timer
        timer.before_step()
        # guarantee non-zero duration
        time.sleep(0.02)
        # records step time if not in warmup
        # if in warmup, it just resets the 
        # step timer and total timer
        timer.after_step(None, None, {})
        trainer._iter = step + 1
    # gets total time including hooks
    # by looking at time - _start_time diff.
    # also gets total time not including hooks
    # by looking at _total_timer. the diff
    # between these two is the time spent in hooks
    # which is also recorded
    timer.after_train()

    # step_time should be logged only for steps >= warmup
    step_times = rec.get_step_scalars().get("step_time", [])
    assert len(step_times) == max(0, 5 - warmup), \
        f"expected {5 - warmup} step_time records, got {len(step_times)}"
    for v, *_ in step_times:
        assert 0.015 <= v <= 0.04

    # ------------------------------------------------------------------ #
    # VALIDATION simulation (3 val steps)
    # ------------------------------------------------------------------ #
    
    timer.before_validation()
    for vstep in range(3):
        timer.before_val_step()
        time.sleep(0.1)
        timer.after_val_step(None, None, {})
        trainer._val_iter = vstep + 1

    time.sleep(0.1) # pretend some extra val work
    timer.after_validation()

    val_step_times = rec.get_step_scalars().get("val_step_time", [])
    assert len(val_step_times) == 3 and all(val > 0 for val, *_ in val_step_times)
    for v, *_ in val_step_times:
        assert 0.05 <= v <= 0.2

    val_time = rec.get_epoch_scalars().get("val_time", [])
    assert len(val_time) == 1 and val_time[0][0] > 0
    assert (0.05*3 + 0.1) <= val_time[0][0] <= (0.2*3 + 0.1), \
        f"Expected validation time to be between 0.25 and 0.7, got {val_time[0][0]}"

    # ------------------------------------------------------------------ #
    # TEST simulation (4 test steps)
    # ------------------------------------------------------------------ #
    
    # reset counters for clean test section
    trainer._iter = 0
    timer.before_test()
    for tstep in range(4):
        timer.before_test_step()
        time.sleep(0.15)
        timer.after_test_step(None, None, {})
        trainer._iter = tstep + 1
    timer.after_test()

    test_step_times = rec.get_step_scalars().get("test_step_time", [])
    
    # warm-up also applies here
    assert len(test_step_times) == max(0, 4 - warmup)
    assert all(0.1 <= val <= 0.2 for val, *_ in test_step_times)

    # ------------------------------------------------------------------ #
    # EPOCH timing
    # ------------------------------------------------------------------ #
    
    timer.before_epoch()
    time.sleep(0.15)
    timer.after_epoch()
    epoch_time = rec.get_epoch_scalars().get("epoch_time", [])
    assert len(epoch_time) == 1 and epoch_time[0][0] > 0
    assert 0.1 <= epoch_time[0][0] <= 0.3, \
        f"Expected epoch time to be between 0.1 and 0.3, got {epoch_time[0][0]}"

    # ---- ---- ---- PeriodicWriter tests ---- ---- ----
    
    local_writer = periodic._writers.writers[0]
    assert isinstance(local_writer, LocalEventWriter), \
        "Expected LocalEventWriter for testing PeriodicWriter"

    # inject dummy scalars
    trainer._iter  = 0
    trainer._epoch = 0
    rec = trainer.event_recorder
    rec.put_scalar("loss", 1.23, scope="step")
    rec.put_scalar("val_metric", 0.90, scope="epoch")

    # trigger the write & clear logic
    periodic.after_epoch()

    # recorder should now be empty
    assert all(len(v) == 0 for v in rec.get_step_scalars().values())
    assert all(len(v) == 0 for v in rec.get_epoch_scalars().values())

    # validate written CSV contents 
    if process_rank() == 0:
        step_csv = local_writer.step_scalars_savepath
        epoch_csv = local_writer.epoch_scalars_savepath

        if not (step_csv.exists() and epoch_csv.exists()):
            raise FileNotFoundError(
                f"Expected step CSV at {step_csv} and epoch CSV at {epoch_csv}, "
                "but one or both are missing."
            )
        else:
            step_df  = pd.read_csv(step_csv)
            epoch_df = pd.read_csv(epoch_csv)

            assert abs(step_df.loc[0, "loss"] - 1.23) < 1e-6
            assert step_df.loc[0, "iter"]  == 0
            assert step_df.loc[0, "epoch"] == 0
            
            assert abs(epoch_df.loc[0, "val_metric"] - 0.90) < 1e-6
            assert epoch_df.loc[0, "epoch"] == 0

    barrier()

    # ---- ---- ---- TorchMemoryStats tests ---- ---- ----

    # make it trigger every step/epoch
    mem_hook._step_period  = 1
    mem_hook._epoch_period = 1

    logdir = mem_hook._logdir
    step_recorder = trainer.event_recorder

    # ------------------------------------------------------------------ #
    # TRAIN STEP
    # ------------------------------------------------------------------ #
    
    trainer._iter = 0
    # allocate a bit of GPU mem so numbers are non-zero
    _ = torch.empty((1024, 1024), device="cuda")
    mem_hook.after_step(None, None, {})
    step_scalars = step_recorder.get_step_scalars()

    keys_ok = all(k in step_scalars for k in
                  ("allocated_mem", "reserved_mem",
                   "max_allocated_mem", "max_reserved_mem")
                )
    
    if not keys_ok:
        raise ValueError(
            "TorchMemoryStats hook did not log expected step data. "
        )

    # ------------------------------------------------------------------ #
    # EPOCH END  (epoch == 0)
    # ------------------------------------------------------------------ #
    
    trainer._epoch = 0
    mem_hook.after_epoch()
    barrier()
    epoch_log_file = logdir / "0.log"
    # check if the epoch log file exists and contains data
    epoch_ok = epoch_log_file.exists() and epoch_log_file.stat().st_size > 0

    if not epoch_ok:
        raise ValueError(
            "TorchMemoryStats hook did not log expected epoch data. "
        )

    # ------------------------------------------------------------------ #
    # TEST STEP
    # ------------------------------------------------------------------ #
    
    trainer._iter = 0
    _ = torch.empty((512, 512), device="cuda")
    mem_hook.after_test_step(None, None, {})
    test_scalars = step_recorder.get_step_scalars()
    test_keys_ok = all(k in test_scalars for k in
                       ("allocated_mem", "reserved_mem", 
                        "max_allocated_mem", "max_reserved_mem")
                    )

    if not test_keys_ok:
        raise ValueError(
            "TorchMemoryStats hook did not log expected test step data. "
        )

    # ------------------------------------------------------------------ #
    # TEST END
    # ------------------------------------------------------------------ #
    
    mem_hook.after_test()
    barrier()
    test_log_file = logdir / "test" / "memory_test.log"
    test_ok = test_log_file.exists() and test_log_file.stat().st_size > 0

    if not test_ok:
        raise ValueError(
            "TorchMemoryStats hook did not log expected test data. "
        )

    # ---- ---- ---- BestMetricSaver tests ---- ---- ----

    metric_name = saver.metric_name
    recorder = trainer.event_recorder

    def _log_epoch_scalar(val):
        recorder._iter = trainer._iter
        recorder._epoch = trainer._epoch
        recorder.put_scalar(metric_name, val, scope="epoch")

    # ------------------------------------------------------------------ #
    # After Validation
    # ------------------------------------------------------------------ #
    
    trainer.best_metric = float("inf")
    trainer._iter, trainer._epoch = 0, 0
    _log_epoch_scalar(0.9)
    saver.after_validation()
    assert abs(trainer.best_metric - 0.9) < 1e-6

    # we set min in config so lower is better
    # next validation should not change the best metric
    _log_epoch_scalar(1.2)
    saver.after_validation()
    assert abs(trainer.best_metric - 0.9) < 1e-6

    # ------------------------------------------------------------------ #
    # After Epoch
    # ------------------------------------------------------------------ #

    # we set eval_after_validation to False
    # so the saver updates the best metric
    # after the epoch ends instead of after validation
    saver.eval_after_validation = False
    trainer._epoch = saver.period - 1
    _log_epoch_scalar(0.8)
    saver.after_epoch()
    assert abs(trainer.best_metric - 0.8) < 1e-6

    # ------------------------------------------------------------------ #
    # After Test
    # ------------------------------------------------------------------ #

    trainer._epoch += 1
    _log_epoch_scalar(0.75)
    saver.after_test()
    assert abs(trainer.best_metric - 0.75) < 1e-6

    # ---- ---- ---- TorchProfiler tests ---- ---- ----

    prof_hook._wait = 0
    prof_hook._warmup = 0
    prof_hook._active = 2
    prof_hook._repeat = 1
    prof_hook.profile_times = 2

    prof_hook.before_train()

    for it in range(2):
        trainer._iter = it
        _ = torch.randn(256, 256) @ torch.randn(256, 256)
        prof_hook.after_step()

    success = True
    if not prof_hook._closed or prof_hook._profiler is not None:
        success = False

    trace_dir = Path(prof_hook._output_dir) / "log"
    traces = [p for p in trace_dir.rglob("*") if p.is_file()]
    if not traces:
        success = False

    barrier()

    if not success:
        raise ValueError(
            "TorchProfiler did not log expected data. "
            "Check the log directory for details."
        )

    # ---- ---- ---- EarlyStopHook tests ---- ---- ----

    metric = ehook.metric_name
    ehook.patience = 2
    threshold = ehook.stopping_threshold

    def _run_epoch(ep_idx, value):
        trainer._epoch = ep_idx
        trainer._iter = 0
        recorder.put_scalar(metric, value, scope="epoch")
        ehook.after_validation()

    # first value (improvement, wait_count stays 0)
    first_val = 0.5
    _run_epoch(0, first_val)
    assert ehook.wait_count == 0
    assert not getattr(trainer, "stop_training", False)

    # barely changes (less than threshold) -> wait_count increments
    val_1 = first_val + threshold * 0.1
    _run_epoch(1, val_1)
    assert ehook.wait_count == 1
    assert not getattr(trainer, "stop_training", False)

    # again no meaningful improvement -> patience reached
    val_2 = val_1 + threshold * 0.1
    _run_epoch(2, val_2)
    assert ehook.wait_count == 2
    # NOTE: we are not setting not getattr() here
    #       so this only passes if stop_training is 
    #       set to True by the hook
    assert getattr(trainer, "stop_training", False)

    # ---- ---- ---- TESTS DONE ---- ---- ----

    report({"success": True})


def test_hooks(config):
    if not torch.cuda.is_available():
        pytest.skip("No GPUs available for testing")

    with open_dict(config):
        config.experiment_name = "test_hooks"
        config.paths.resume_checkpointdir = None
        config.clusters.worker_nodes = 1
        config.clusters.gpus_per_worker = torch.cuda.device_count()
        config.clusters.cpus_per_gpu = 4
        config.clusters.mem_per_cpu = 31000

    metrics = distributed_test(cfg=config, test="tests.training.test_hooks._test_hooks_dist")
    assert metrics.get("success", False), "Distributed hooks test failed"