'''
Take transcriptions (.txt files) and parses it into 
scribie.com's expected output format.
'''

fn = 'audio1_transcript.txt'
trans_file = 'transcriptions/' + fn
f = open(trans_file, "r")

transcription = f.readlines()

transcription = [t.replace("\n", "") for t in transcription]

output = ""
for t in transcription:
    if t != '' or "SPEAKER" not in t:
        # output += t + "\n"
        print("SPEAKER")
        print(repr(t))
        print("****")

print(output)