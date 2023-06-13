import json

# Set the desired filename
output_filename = "SDv14.json"

with open("dataset/data/test_data.json") as f:
    my_dict = [json.loads(line) for line in f]

    output_data = []

    for i in my_dict:
        description_1 = i['caption1_modified']
        description_2 = i['caption2_modified']
        with open('badwords.txt', 'r') as f:
            for line in f:
                for word in line.split(","):
                    description_1 = description_1.replace(word, '*'*len(word))
                    description_2 = description_2.replace(word, '*'*len(word))

        data = {
            "img_local_path": i['img_local_path'],
            "original_caption1": i['caption1'],
            "caption1_mod": description_1,
            "img_gen1": f"SDv14-GeneratedDataset/{i['img_local_path'].translate({ord(n): None for n in '.jpng/test'})}gen1.jpg",
            "original_caption2": i['caption2'],
            "caption2_mod": description_2,
            "img_gen2": f"SDv14-GeneratedDataset/{i['img_local_path'].translate({ord(n): None for n in '.jpng/test'})}gen2.jpg",
            "label": i['context_label']
        }

        output_data.append(data)

# Serialize the data to JSON format and save it to the output file
with open(output_filename, 'w') as outfile:
    json.dump(output_data, outfile)

print(f"Data saved to {output_filename}")
