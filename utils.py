import json


def read_json_file(path, jsonl=True):
    with open(path, encoding='utf-8') as input_stream:
        if jsonl:
            data = [json.loads(line) for line in input_stream]
        else:
            data = json.loads(input_stream.read())
        return data


def save_json_file(data, path, jsonl=True):
    with open(path, 'w', encoding='utf-8') as output_stream:
        if jsonl:
            output_str = [json.dumps(single_example) for single_example in data]
            output_str = '\n'.join(output_str)
        else:
            output_str = json.dumps(data)
        output_stream.write(output_str)