from datasets import load_dataset


dataset = load_dataset('facebook/winoground', use_auth_token='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')['test']

id = 14
caption_0 = dataset[id]['caption_0']
caption_1 = dataset[id]['caption_1']
img_0 = dataset[id]['image_0']
img_1 = dataset[id]['image_1']

img_0.save('/user_data/junruz/wino/img_0_14.png')
img_1.save('/user_data/junruz/wino/img_1_14.png')

print(caption_0)
print(caption_1)