targets_dir="./test-targets"
tests_dir="./tests"
for eachfile in $(ls $targets_dir)
do
    diff_output="$(python3.9 main.py < $tests_dir/$eachfile | diff $targets_dir/$eachfile -)"
    if [ -z "$diff_output" ]
    then
      echo "$tests_dir/$eachfile\t ok"
    else
      echo "$tests_dir/$eachfile\t FAILED"
    fi
done