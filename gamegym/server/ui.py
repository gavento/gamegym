import xml.etree.ElementTree as et
import collections

"""
def element(name, text=None, **kw):
    e = et.Element(name)
    for k, v in sorted(kw.items()):
        if v is not None:
            e.set(k.replace("_", "-"), str(v))
    if text is not None:
        e.text = str(text)
    return e


def add_element(parent, name, text=None, **kw):
    e = element(name, text=text, **kw)
    parent.append(e)
    return e
"""


class BuildContext:

    def __init__(self, callback_prefix=""):
        self.callbacks = []
        self.callback_prefix = callback_prefix

    def register_callback(self, callback):
        callback_id = self.callback_prefix + str(len(self.callbacks))
        self.callbacks.append(callback)
        return callback_id


class Element:

    def __init__(self, name, text=None, **kw):
        self.name = name
        self.text = text
        self.kw = kw
        self.childs = []

    def add(self, name, text=None, **kw):
        e = Element(name, text, **kw)
        self.childs.append(e)
        return e

    def to_xml(self, ctx):
        e = et.Element(self.name)
        for k, v in sorted(self.kw.items()):
            if v is None:
                continue
            if callable(v):
                callback_id = ctx.register_callback(v)
                value = "window.location.href='{}'".format(callback_id)
            else:
                value = str(v)
            e.set(k.replace("_", "-"), value)
        if self.text is not None:
            e.text = str(self.text)
        for child in self.childs:
            e.append(child.to_xml(ctx))
        return e


class CardBuilder:

    def __init__(self, width=80, height=120, border_color="black", fill_color="gray"):
        self.width = width
        self.height = height
        self.border_color = border_color
        self.fill_color = fill_color
        self.font_family = None

    def build(self, parent, x, y, text, callback=None):
        parent.add("rect", x=x, y=y,
                   width=self.width, height=self.height,
                   rx=15, ry=15,
                   fill=self.fill_color,
                   stroke=self.border_color, stroke_width=7)

        if text is not None:
            parent.add("text", text=text,
                       x=x + self.width / 2, y=y + self.height / 2,
                       fill="white", font_size=40, text_anchor="middle",
                       alignment_baseline="middle", font_family=self.font_family)

        if callback:
            parent.add("rect", x=x, y=y,
                    width=self.width, height=self.height,
                    style="fill-opacity: 0; stroke-opacity: 0; cursor: pointer",
                    onclick=callback)


class Screen(Element):

    def __init__(self, width, height):
        super().__init__("svg", width=width, height=height)
