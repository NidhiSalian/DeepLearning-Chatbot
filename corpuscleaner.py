from sys import argv

'''Removes numbers, the word "Chapter", and any additional specified
strings from a given text file.'''

script, source = argv

source_file = open(source)

source_text = source_file.read()

source_file.close()

# to_remove = "Unique annoying text that appears frequently in this corpus"
new_source = source_text.translate(None, "0123456789_")
new_source = new_source.replace("GUTIERREZ:", "")
new_source = new_source.replace("LOPEZ:", "")
new_source = new_source.replace("JONE", "")
new_source = new_source.replace("JOHNSTON:", "")
new_source = new_source.replace("SCHRODER:", "")
new_source = new_source.replace("BROWN:", "")
new_source = new_source.replace("JACKSON:", "")
new_source = new_source.replace("MORIATY:", "")
new_source = new_source.replace("BAYN", "")
new_source = new_source.replace("KASSON:", "")
new_source = new_source.replace("LEVIN ", "")
new_source = new_source.replace("HERSHEY:", "")
new_source = new_source.replace("CONOVER:", "")
new_source = new_source.replace("ANDREW", "")
new_source = new_source.replace("LENSING:", "")
new_source = new_source.replace("SCHLENVOG:", "")
new_source = new_source.replace("KIFER:", "")
new_source = new_source.replace("LEAVY:", "")
# new_source = new_source.replace(to_remove, "")


source_file = open(source, "w")

source_file.write(new_source)

source_file.close()
