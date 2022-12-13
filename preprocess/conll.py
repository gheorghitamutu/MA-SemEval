
def remove_comments_from_conll_file(source, destination):
    with open(source, 'r', encoding='utf-8') as f:
        with open(destination, 'w', encoding='utf-8') as g:
            for line in f.readlines():
                if not line.startswith('#'):
                    g.write(line)
