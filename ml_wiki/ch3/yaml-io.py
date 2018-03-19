import yaml
# 파이썬 데이터를 YAML 데이터로 출력하기
customer = [
    { "name": "InSeong", "age": "24", "gender": "man" },
    { "name": "Akatsuki", "age": "22", "gender": "woman" },
    { "name": "Harin", "age": "23", "gender": "man" },
    { "name": "Yuu", "age": "31", "gender": "woman" }
]
# 파이썬 데이터를 YAML 데이터로 변환하기
yaml_str = yaml.dump(customer)
print(yaml_str)
print("--- --- ---")
# YAML 데이터를 파이썬 데이터로 변환하기
data = yaml.load(yaml_str)
# 이름 출력하기
for p in data:
    print(p["name"])