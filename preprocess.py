from sentence_splitter import SentenceSplitter, split_text_into_sentences
from pathlib import Path
import sys

data_cat = "train"
if len(sys.argv) >= 2:
	if sys.argv[1] == "dev":
		data_cat = "dev"

print("Preprocessing: ", data_cat)


splitter = SentenceSplitter(language='en')

folder = "./preprocessed-data/"

src_files = Path(folder).glob("*"+data_cat)

files = [x for x in src_files]

# Gutenberg
guten = files[5].read_text(encoding="utf-8").splitlines()
cur_text = ""
with open("preprocessed-data/gutenberg."+data_cat, 'w') as out:
	for line in guten:
		if line == '':
			if len(cur_text) > 0:
				out.write('\n'.join(splitter.split(text=cur_text))+'\n')
				cur_text = ""
		else:
			if len(cur_text) > 0:
				cur_text += ' '
			cur_text += line
	if len(cur_text) > 0:
		out.write('\n'.join(splitter.split(text=cur_text))+'\n')
	

# qed
qed = files[3].read_text(encoding="utf-8").splitlines()
with open("preprocessed-data/qed."+data_cat, "w") as out:
	for line in qed:
		out.write(line.lower()+"\n")

# children_stories

c_stories = files[7].read_text(encoding="utf-8").splitlines()
with open("preprocessed-data/children_stories."+data_cat, "w") as out:
	for line in c_stories:
		out.write('\n'.join(splitter.split(text=line))+'\n')


# simple_wikipedia

s_wiki = files[9].read_text(encoding="utf-8").splitlines()
with open("preprocessed-data/simple_wikipedia."+data_cat, "w") as out:
	for line in s_wiki:
		out.write('\n'.join(splitter.split(text=line))+'\n')




# wikipedia

wiki = files[4].read_text(encoding="utf-8").splitlines()
with open("preprocessed-data/wikipedia."+data_cat, "w") as out:
	for line in wiki:
		out.write('\n'.join(splitter.split(text=line))+'\n')




