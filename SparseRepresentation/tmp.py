'''
Created on 2014/3/25

@author: bhchen
'''
dict_list = [{'price': 99, 'barcode': '2342355'}, {'price': 88, 'barcode': '2345566'}, {'price': 77, 'barcode': '2342377'}]\

max_priced_item = max(dict_list, key=lambda x:x['price'])
min_priced_item = min(dict_list, key=lambda x:x['price'])
print(max_priced_item, min_priced_item)