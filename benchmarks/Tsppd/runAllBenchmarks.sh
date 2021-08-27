# 1 mins
python3 runBenchmarks.py -s mdd -t  60 -q 50000 --wc   10 --mc 1000 --wg 0 --mg     0 --eq 0.60 --neq 0.01;
python3 runBenchmarks.py -s mdd -t  60 -q 50000 --wc  100 --mc  100 --wg 0 --mg     0 --eq 0.60 --neq 0.01;
python3 runBenchmarks.py -s mdd -t  60 -q 50000 --wc 1000 --mc   10 --wg 0 --mg     0 --eq 0.60 --neq 0.01;
python3 runBenchmarks.py -s mdd -t  60 -q 50000 --wc   10 --mc 1000 --wg 3 --mg 25000 --eq 0.60 --neq 0.01;
python3 runBenchmarks.py -s mdd -t  60 -q 50000 --wc  100 --mc  100 --wg 3 --mg 25000 --eq 0.60 --neq 0.01;
python3 runBenchmarks.py -s mdd -t  60 -q 50000 --wc 1000 --mc   10 --wg 3 --mg 25000 --eq 0.40 --neq 0.01;

# 5 mins
python3 runBenchmarks.py -s mdd -t 300 -q 50000 --wc   10 --mc 1000 --wg 0 --mg     0 --eq 0.60 --neq 0.01;
python3 runBenchmarks.py -s mdd -t 300 -q 50000 --wc  100 --mc  100 --wg 0 --mg     0 --eq 0.60 --neq 0.01;
python3 runBenchmarks.py -s mdd -t 300 -q 50000 --wc 1000 --mc   10 --wg 0 --mg     0 --eq 0.60 --neq 0.01;
python3 runBenchmarks.py -s mdd -t 300 -q 50000 --wc   10 --mc 1000 --wg 3 --mg 25000 --eq 0.60 --neq 0.01;
python3 runBenchmarks.py -s mdd -t 300 -q 50000 --wc  100 --mc  100 --wg 3 --mg 25000 --eq 0.60 --neq 0.01;
python3 runBenchmarks.py -s mdd -t 300 -q 50000 --wc 1000 --mc   10 --wg 3 --mg 25000 --eq 0.60 --neq 0.01;

