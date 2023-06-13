import json

def SDjson():
    # Set the desired filename
    output_filename = "SDv14.json"

    with open("dataset/data/test_data.json") as f:
        my_dict = [json.loads(line) for line in f]

        with open(output_filename, 'w') as outfile:
            for i, data_dict in enumerate(my_dict, start=1):
                description_1 = data_dict['caption1_modified']
                description_2 = data_dict['caption2_modified']
                with open('badwords.txt', 'r') as f:
                    for line in f:
                        for word in line.split(","):
                            description_1 = description_1.replace(word, '*'*len(word))
                            description_2 = description_2.replace(word, '*'*len(word))

                data = {
                    "img_local_path": data_dict['img_local_path'],
                    "original_caption1": data_dict['caption1'],
                    "caption1_mod": description_1,
                    "img_gen1": f"SDv14-GeneratedDataset/{data_dict['img_local_path'].translate({ord(n): None for n in '.jpng/test'})}gen1.jpg",
                    "original_caption2": data_dict['caption2'],
                    "caption2_mod": description_2,
                    "img_gen2": f"SDv14-GeneratedDataset/{data_dict['img_local_path'].translate({ord(n): None for n in '.jpng/test'})}gen2.jpg",
                    "label": data_dict['context_label']
                }

                # Serialize the individual data to JSON format and write it to the output file
                json_data = json.dumps(data)
                outfile.write(json_data + "\n")

    print(f"Data saved to {output_filename}")
