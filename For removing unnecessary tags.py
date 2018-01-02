import filemapper as fm
all_files = fm.load('d065j')
for i in range(len(all_files)):
    f1 = all_files[i]
    f2 = all_files[i] +".txt"
    with open(f1) as infile, open(f2, 'w') as outfile:
        copy = False
        for line in infile:
            if line.strip() == "<TEXT>":
                copy = True
            elif line.strip() == "</TEXT>":
                copy = False
            elif copy:
                outfile.write(line)

