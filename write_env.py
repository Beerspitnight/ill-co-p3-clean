with open("clean.env", "r") as source:
    contents = source.read()

with open("/tmp/clean.env", "w") as dest:
    dest.write(contents)

print("✅ clean.env copied to /tmp!")
