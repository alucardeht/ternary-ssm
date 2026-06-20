import sys

content = open('train.c', 'r', encoding='utf-8').read()

old_str = '''        } else if (strcmp(argv[i], "--martingale-lambda") == 0 && i + 1 < argc) {
            martingale_lambda = (float)atof(argv[++i]);'''

new_str = old_str + '''
        } else if (strcmp(argv[i], "--continuous-ternary-margin-gate") == 0) {
            continuous_ternary_margin_enabled = 1;
        } else if (strcmp(argv[i], "--continuous-ternary-margin-c") == 0 && i + 1 < argc) {
            continuous_ternary_margin_c = (float)atof(argv[++i]);'''

if old_str in content:
    content = content.replace(old_str, new_str)
    
    old_str2 = '''    float martingale_lambda = 0.0f;'''
    new_str2 = old_str2 + '''
    int continuous_ternary_margin_enabled = 0;
    float continuous_ternary_margin_c = 0.1f;'''
    
    if old_str2 in content:
        content = content.replace(old_str2, new_str2)
        
        old_str3 = '''    m->martingale_lambda = martingale_lambda;'''
        new_str3 = old_str3 + '''
    m->continuous_ternary_margin_enabled = continuous_ternary_margin_enabled;
    m->continuous_ternary_margin_c = continuous_ternary_margin_c;'''
        
        if old_str3 in content:
            content = content.replace(old_str3, new_str3)
            open('train.c', 'w', encoding='utf-8', newline='\n').write(content)
            print("Success")
        else:
            print("old_str3 not found")
            sys.exit(1)
    else:
        print("old_str2 not found")
        sys.exit(1)
else:
    print("old_str not found")
    sys.exit(1)
