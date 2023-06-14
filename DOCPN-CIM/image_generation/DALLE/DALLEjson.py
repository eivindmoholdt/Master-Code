import json
from config import COSMOS_DATA
import os


def dallejson():
    # Set the desired filename
    output_filename = "DALLE.json"

    with open(os.path.join(COSMOS_DATA, 'test_data.json')) as f:
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
                    "img_local_path": i['img_local_path'],
                    "original_caption1":i['caption1'],
                    "caption1_mod": description_1,
                    "img_gen1": f"DallE-GeneratedDataset/images/{i['img_local_path'].translate( { ord(n): None for n in '.jpng/test'} )}_gen1.png",
                    "original_caption2":i['caption2'],
                    "caption2_mod": description_2,
                    "img_gen2": f"DallE-GeneratedDataset/images/{i['img_local_path'].translate( { ord(n): None for n in '.jpng/test'} )}_gen2.png",
                    "label": i['context_label']
                    }

                # Serialize the individual data to JSON format and write it to the output file
                json_data = json.dumps(data)
                outfile.write(json_data + "\n")

    print(f"Data saved to {output_filename}")



