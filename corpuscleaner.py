from sys import argv

'''Removes numbers, the word "Chapter", and any additional specified
strings from a given text file.'''

script, source = argv

source_file = open(source)

source_text = source_file.read()

source_file.close()

# to_remove = "Unique annoying text that appears frequently in this corpus"
new_source = source_text.translate(None, "0123456789_")
new_source = new_source.replace("Chapter", "")
new_source = new_source.replace("Tape", "")
new_source = new_source.replace("Call", "")

# new_source = new_source.replace(to_remove, "")


source_file = open(source, "w")

source_file.write(new_source)

source_file.close()
