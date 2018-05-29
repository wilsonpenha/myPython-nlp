import RAKE as rake  
import operator

rake_object = rake.Rake("SmartStoplist.txt")

sample_file = open("/data/corpus/Canada Stock Market Terms.txt", 'r', encoding="utf-8")
text = sample_file.read()

keywords = rake_object.run(text,1,3,1)

# 3. print results
for key, score in keywords:
    print("Keywords:", key, " ==>> Score:", score)