# How to run
## run all available testcases
```
./reprounzip directory setup ch4109_hm2665.rpz ./result
cd ./result/root/home/ch4109/ADB/ADB-RepCRec
chmod +x run.sh
./run.sh
```

## run new test case
```
./reprounzip directory setup ch4109_hm2665.rpz ./result
cd ./result/root/home/ch4109/ADB/ADB-RepCRec
python3.7 main.py < [test_file]
```

# Design
See design-doc.pdf