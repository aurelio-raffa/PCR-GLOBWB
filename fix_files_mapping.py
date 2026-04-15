import os
import re
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--inp_ini')
parser.add_argument('--inp_dir')
parser.add_argument('--data_ph')
parser.add_argument('--out_txt')
parser.add_argument('--out_ini')

if __name__ == "__main__":
    args = parser.parse_args()
    file_regex = fr'{args.inp_dir}/(.*)\.(.*)'

    with open(args.inp_ini, 'r') as handle:
        contents = handle.read()

    links = re.findall(file_regex, contents, re.MULTILINE)

    unchanged = set()
    mapped = {}
    name_conflict = {}
    unmapped = {}
    for pth, ext in links:
        full_pth = os.path.join(args.data_ph, args.inp_dir, f'{pth}.{ext}')
        dir_pth = os.path.join(args.data_ph, args.inp_dir, *os.path.split(pth)[:-1])
        if os.path.exists(full_pth):    # file is ok
            unchanged.add(pth)
        elif re.search(r'%([0-9]+)[di]', str(full_pth)):  # check if wildcard is used
            matches = glob.glob(full_pth.replace(re.search(r'%([0-9]+)[di]', str(full_pth)).group(0), '*'))
            if len(matches) == 0:
                unmapped[full_pth] = 'no matching wildcard found'
            elif len(matches) == 1:
                unchanged.add(full_pth)
            else:
                name_conflict[full_pth] = [os.path.join(str(dir_pth), f) for f in matches]
        elif os.path.isdir(dir_pth):    # at least the directory exists
            if sum(
                    [
                        int(os.path.splitext(f)[0] == os.path.split(pth)[-1])
                        for f in os.listdir(str(dir_pth))
                    ]
            ):
                # a single file with the same name exists but with different ext
                [mapped[full_pth]] = [
                    os.path.join(str(dir_pth), f) for f in os.listdir(str(dir_pth))
                    if int(os.path.splitext(f)[0] == os.path.split(pth)[-1])
                ]
            elif any(
                    [
                        os.path.splitext(f)[0] == os.path.split(pth)[-1]
                        for f in os.listdir(str(dir_pth))
                    ]
            ):
                # multiple different versions exist...
                name_conflict[full_pth] = [
                    os.path.join(str(dir_pth), f) for f in os.listdir(str(dir_pth))
                    if os.path.splitext(f)[0] == os.path.split(pth)[-1]
                ]
            else:
                # no file exists with same name before extension
                unmapped[full_pth] = 'no file of this name (net of extension) exists in directory'
        else:
            # not even the directory exits
            unmapped[full_pth] = 'parent directory of file does not exists'

    # create and print summary
    s = f'Of the {len(links)} paths in under {args.inp_dir}/ in {args.inp_ini}:\n'
    s += f'-> {len(unchanged)} exact matches found;\n'
    s += f'-> {len(mapped)} found under a different extension;\n'
    s += f'-> {len(name_conflict)} have multiple incompatible extensions;\n'
    s += f'-> {len(unmapped)} have not been found under any extension.\n'
    s += '\nRecovered File Mapping:\n'
    for k, v in mapped.items():
        s += f'- {k}: {v}\n'
    s += '\nPossible Additional Mappings:\n'
    for k, v in name_conflict.items():
        s += f'- {k}:\n'
        s += '\n'.join([f'\t- {e}' for e in v]) + '\n'
    s += '\nNot found:\n'
    for k, v in unmapped.items():
        s += f'- {k}\t\t\t# {v}\n'

    # print and save
    print(s)
    with open(args.out_txt, 'w') as handle:
        handle.write(s)