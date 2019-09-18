import json
import codecs


def load_configs(file='./tmp/input/configs.json'):
    object_text = codecs.open(file, 'r', encoding='utf-8').read()
    return json.loads(object_text)


def save_configs(configs, file='/tmp/input/configs.json'):
    json.dump(configs, open(file, 'w'))
