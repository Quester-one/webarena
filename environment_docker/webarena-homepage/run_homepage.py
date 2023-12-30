import sys
sys.path.append('/data/mentianyi/code/webarena')
from config_private import SHOPPING, SHOPPING_ADMIN, REDDIT, GITLAB, MAP, WIKIPEDIA, HOMEPAGE_host, HOMEPAGE_port
import os
import shutil
import subprocess


def copy_html_file(source_path, destination_path):
    try:
        shutil.copy(source_path, destination_path)
        print(f"成功复制文件从 {source_path} 到 {destination_path}")
    except FileNotFoundError:
        print("指定的源文件不存在。")
    except PermissionError:
        print("没有足够的权限复制文件。")
    except Exception as e:
        print(f"发生错误: {e}")


def replace_hostname_in_html(file_path, raw_hostname, actual_hostname):
    perl_command = [
        "perl",
        "-pi",
        "-e",
        f's|{raw_hostname}|{actual_hostname}|g',
        file_path
    ]
    try:
        subprocess.run(perl_command, check=True)
        print("Perl 命令执行成功！")
    except subprocess.CalledProcessError as e:
        print(f"Perl 命令执行失败：{e}")


if __name__ == "__main__":
    copy_html_file("templates/index.html", "templates/index_temp.html")
    replace_hostname_in_html("templates/index_temp.html", "raw_SHOPPING_ADMIN", SHOPPING_ADMIN)
    replace_hostname_in_html("templates/index_temp.html", "raw_SHOPPING", SHOPPING)
    replace_hostname_in_html("templates/index_temp.html", "raw_REDDIT", REDDIT)
    replace_hostname_in_html("templates/index_temp.html", "raw_GITLAB", GITLAB)
    replace_hostname_in_html("templates/index_temp.html", "raw_MAP", MAP)
    replace_hostname_in_html("templates/index_temp.html", "raw_WIKIPEDIA", WIKIPEDIA)
    try:
        server = subprocess.Popen("FLASK_APP=app.py flask run --host={} --port={}".format(HOMEPAGE_host,HOMEPAGE_port), shell=True)
        print("Homepage开启成功！")
    except subprocess.CalledProcessError as e:
        print(f"Perl 命令执行失败：{e}")
