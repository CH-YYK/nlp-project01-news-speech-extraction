def split(sentence: str) ->'List(str)':
    '”“'
    # merge two quotations
    sentence = sentence.replace('”“', '，')
    N = len(sentence)

    res = []
    stack = []
    l = 0
    for i, char in enumerate(sentence):
        if char == '“':
            stack.append(i)
            continue
        if char == "。" or char == "？" or char == "！":
            if not stack:
                res.append(sentence[l:i])
                l = i+1
        if char == "”":
            if sentence[i-1] == '。': # direct quotation
                stack.pop()
                res.append(sentence[l:i+1])
            elif i + 1 < N and sentence[i+1] == '。': # direct quotation
                stack.pop()
                res.append(sentence[l:i+1])
            l = i+1
    return res






