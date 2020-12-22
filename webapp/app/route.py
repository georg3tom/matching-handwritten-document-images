from flask import jsonify, render_template, request, Blueprint, send_from_directory
from hwmatcher.search import stringQuery

routes = Blueprint("routes", __name__)
IMG_DIR = './static/images/'


@routes.route("/")
def root():
    """
    Main page
    """
    return render_template("main.html")


@routes.route("/searchString", methods=["POST"])
def searchString():
    """
    Queries for nearest neighbors
    """
    string = request.form['string']
    obj = stringQuery()
    savePath = './webapp/app/static/images/'
    images, distances = obj.query(string, savePath)
    ret = jsonify(
        images=list(images),
        votes=list([0]*(len(images))),
    )
    return ret


@routes.route("/static/image/<path:filename>")
def static_images(filename):
    return send_from_directory(IMG_DIR, filename)
