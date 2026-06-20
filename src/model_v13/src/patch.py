import sys

content = open('train.c', 'r', encoding='utf-8').read()

old_str = '''        } else if (strcmp(argv[i], "--amortized-ce-martingale-min-descent-per-paid-work") == 0 && i + 1 < argc) {
            amortized_ce_martingale_min_descent_per_paid_work = (float)atof(argv[++i]);'''

new_str = old_str + '''
        } else if (strcmp(argv[i], "--martingale-lambda") == 0 && i + 1 < argc) {
            martingale_lambda = (float)atof(argv[++i]);'''

if old_str in content:
    content = content.replace(old_str, new_str)
    
    old_str2 = '''    m->amortized_ce_martingale_min_descent_per_paid_work =
        amortized_ce_martingale_min_descent_per_paid_work;'''
    new_str2 = old_str2 + '''
    m->martingale_lambda = martingale_lambda;'''
    
    if old_str2 in content:
        content = content.replace(old_str2, new_str2)
        open('train.c', 'w', encoding='utf-8', newline='\n').write(content)
        print("Success")
    else:
        print("old_str2 not found")
        sys.exit(1)
else:
    print("old_str not found")
    sys.exit(1)
