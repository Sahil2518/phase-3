with open('src/demo_task24.py', 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('    \u2713 {s}', '    [*] {s}')
with open('src/demo_task24.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed unicode checkmark')
