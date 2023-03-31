
#Borrowed from COSMOS source code
#https://github.com/sanonymous22/COSMOS

def modify_caption_replace_entities(caption_text):
    """
        Utility function to replace named entities in the caption with their corresponding hypernyms

        Args:
            caption_text (str): Original caption with named entities

        Returns:
            caption_modified (str): Modified caption after replacing named entities
    """
    doc = nlp(caption_text)
    caption_modified = caption_text
    caption_entity_list = []
    for ent in doc.ents:
        caption_entity_list.append((ent.text, ent.label_))
        caption_modified = caption_modified.replace(ent.text, ent.label_, 1)
    return caption_modified
