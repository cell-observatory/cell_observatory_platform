export LC_ALL=C.UTF-8
export LANG=C.UTF-8

while getopts ":a:r:" option;do
    case "${option}" in
    a) a=${OPTARG}
       cluster_address=$a
       echo cluster_address=$cluster_address
    ;;
    r) r=${OPTARG}
       required=$r
       echo required=$required
    ;;
    *) echo "Did not supply the correct arguments"
    ;;
    esac
done

TIMEOUT=60 # seconds
check="ray status --address $cluster_address"
echo "Checking status for $cluster_address"
st="$(date -u +%s)"

while true; do
  # from https://stackoverflow.com/questions/12321469/retry-a-bash-command-with-timeout
  ct="$(date -u +%s)"
  elapsed=$(( $ct - $st ))

  if [ $elapsed -gt $TIMEOUT ]; then
      echo "Timeout after $TIMEOUT seconds"
      ray stop --force
      exit 1
  fi

  out=$($check)
  echo $out
  status=$?

  if [ $status -ne 0 ]; then
      echo "Cluster status command failed with exit code $status"
      ray stop --force
      exit 1
  fi

  ready=$(echo "$out" | awk '/Active:/{ f = 1; next } /Healthy:/{ f = 1; next } /Pending:/{ f = 0 } f' | wc -l)
  echo "Cluster has $ready/$required node(s) [waiting for $(( $required - $ready )) node(s)] "

  if [ $required -eq $ready ]; then
      break
  fi

  sleep 1
done

$check
