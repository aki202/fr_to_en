import const

eng_prefixes = (
  "i am ", "i m ",
  "he is", "he s ",
  "she is", "she s ",
  "you are", "you re ",
  "we are", "we re ",
  "they are", "they re "
)

def filterPair(p):
  return len(p[0].split(' ')) < const.MAX_LENGTH \
    and len(p[1].split(' ')) < const.MAX_LENGTH \
    and p[1].startswith(eng_prefixes)

def filterPairs(pairs):
  return [pair for pair in pairs if filterPair(pair)]
