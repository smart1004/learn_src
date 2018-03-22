import yaml
# 문자열로 YAML을 정의합니다.
yaml_str = """
# 정의
color_def:
  - &color1 "#FF0000"
  - &color2 "#00FF00"
  - &color3 "#0000FF"
# 별칭 테스트
color:
  title: *color1
  body: *color2
  link: *color3
"""
# YAML 데이터 분석하기
data = yaml.load(yaml_str)
# 별칭이 전개됐는지 테스트하기
print("title=", data["color"]["title"])
print("body=", data["color"]["body"])
print("link=", data["color"]["link"])