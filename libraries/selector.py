# Splits the data into (training, testing)
def split_data(path, percent):
    import json
    import math
    
    with open(path, 'r') as fp:
        all_objs = [json.loads(x) for x in fp.readlines()]
        
    index = math.floor((percent/100)*len(all_objs))
    training = []
    test = []
    for x in all_objs[:index]:
        if x['asin'] not in all_objs[index]['asin']:
            training.append(x)
        else:
            test.append(x)
    
    for x in all_objs[index:]:
        test.append(x)
        
    return (training, test)
