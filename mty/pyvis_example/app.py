from flask import Flask,render_template_string,send_from_directory
from config_private import VISIALIZE_host,VISIALIZE_port,VISIALIZE

app = Flask(__name__)


@app.route("/")
def example() -> str:
    from pyvis.network import Network

    # 创建一个网络对象
    net = Network()

    # 添加节点和边缘
    net.add_node(1, label='Node 9',shape="image",image="{}/img_lights.jpg".format(VISIALIZE))
    net.add_node(2, label='Node 2',shape="image",image="https://www.w3schools.com/w3css/img_lights.jpg")
    net.add_node(3, label='Node 3')
    net.add_edge(1, 2)
    net.add_edge(1, 3)

    # 保存图谱为HTML文件并在浏览器中打开
    net.show('example.html', notebook=False)
    return render_template_string(net.html)

@app.route('/<filename>')
def serve_image(filename):
    images_directory = 'pyvis_example'
    return send_from_directory(images_directory, filename)


if __name__ == "__main__":
    app.run(host=VISIALIZE_host, port=VISIALIZE_port)