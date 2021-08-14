# 1 mins
python3 runBenchmarks.py -s mdd -t  60 -q 10000 --wc   10 --pc 1000 --wg 0 --pg    0 --eq 60 --neq 03;
python3 runBenchmarks.py -s mdd -t  60 -q 10000 --wc  100 --pc  100 --wg 0 --pg    0 --eq 60 --neq 03;
python3 runBenchmarks.py -s mdd -t  60 -q 10000 --wc 1000 --pc   10 --wg 0 --pg    0 --eq 60 --neq 03;
python3 runBenchmarks.py -s mdd -t  60 -q 10000 --wc   10 --pc 1000 --wg 3 --pg 2000 --eq 60 --neq 03;
python3 runBenchmarks.py -s mdd -t  60 -q 10000 --wc  100 --pc  100 --wg 3 --pg 2000 --eq 60 --neq 03;
python3 runBenchmarks.py -s mdd -t  60 -q 10000 --wc 1000 --pc   10 --wg 3 --pg 2000 --eq 60 --neq 03;

# 5 mins
python3 runBenchmarks.py -s mdd -t 300 -q 10000 --wc   10 --pc 1000 --wg 0 --pg    0 --eq 60 --neq 03;
python3 runBenchmarks.py -s mdd -t 300 -q 10000 --wc  100 --pc  100 --wg 0 --pg    0 --eq 60 --neq 03;
python3 runBenchmarks.py -s mdd -t 300 -q 10000 --wc 1000 --pc   10 --wg 0 --pg    0 --eq 60 --neq 03;
python3 runBenchmarks.py -s mdd -t 300 -q 10000 --wc   10 --pc 1000 --wg 3 --pg 2000 --eq 60 --neq 03;
python3 runBenchmarks.py -s mdd -t 300 -q 10000 --wc  100 --pc  100 --wg 3 --pg 2000 --eq 60 --neq 03;
python3 runBenchmarks.py -s mdd -t 300 -q 10000 --wc 1000 --pc   10 --wg 3 --pg 2000 --eq 60 --neq 03;

