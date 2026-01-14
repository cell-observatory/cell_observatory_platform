

class DeepCopyInputsAsTargets:
    def __init__(self):
        pass

    def __call__(self, data: dict) -> dict:
        if "data_tensor" not in data:
            raise KeyError("DeepCopyInputsAsTargets expects 'data_tensor' in dict.")
        if "metainfo" not in data:
            data["metainfo"] = {}
        if "targets" in data["metainfo"] and data["metainfo"]["targets"] is not None:
            raise ValueError("targets already exists in metainfo")
        data["metainfo"]["targets"] = [data["data_tensor"].clone()]
        return data