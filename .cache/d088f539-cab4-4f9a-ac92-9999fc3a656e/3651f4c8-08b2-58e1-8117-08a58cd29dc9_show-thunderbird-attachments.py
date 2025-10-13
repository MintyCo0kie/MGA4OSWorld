import lxml.etree
from lxml.etree import _Element
import pyatspi
from pyatspi import Accessible, StateType
from pyatspi import Component, Document
from pyatspi import Text as ATText
from pyatspi import Value as ATValue
from pyatspi import Action as ATAction

from lxml.cssselect import CSSSelector
import pyautogui
import os.path
import time

from typing import Dict, List
from typing import Any, Optional

_accessibility_ns_map = { "st": "uri:deskat:state.at-spi.gnome.org"
                        , "attr": "uri:deskat:attributes.at-spi.gnome.org"
                        , "cp": "uri:deskat:component.at-spi.gnome.org"
                        , "doc": "uri:deskat:document.at-spi.gnome.org"
                        , "docattr": "uri:deskat:attributes.document.at-spi.gnome.org"
                        , "txt": "uri:deskat:text.at-spi.gnome.org"
                        , "val": "uri:deskat:value.at-spi.gnome.org"
                        , "act": "uri:deskat:action.at-spi.gnome.org"
                        }
def _create_node(node: Accessible) -> _Element:
    attribute_dict: Dict[str, Any] = {"name": node.name}

    #  States {{{ # 
    states: List[StateType] = node.getState().get_states()
    for st in states:
        state_name: str = StateType._enum_lookup[st]
        attribute_dict[ "{{{:}}}{:}"\
                            .format( _accessibility_ns_map["st"]
                                   , state_name.split("_", maxsplit=1)[1].lower()
                                   )
                      ] = "true"
    #  }}} States # 

    #  Attributes {{{ # 
    attributes: List[str] = node.getAttributes()
    for attrbt in attributes:
        attribute_name: str
        attribute_value: str
        attribute_name, attribute_value = attrbt.split(":", maxsplit=1)
        attribute_dict[ "{{{:}}}{:}"\
                            .format( _accessibility_ns_map["attr"]
                                   , attribute_name
                                   )
                      ] = attribute_value
    #  }}} Attributes # 

    #  Component {{{ # 
    try:
        component: Component = node.queryComponent()
    except NotImplementedError:
        pass
    else:
        attribute_dict["{{{:}}}screencoord".format(_accessibility_ns_map["cp"])] = str(component.getPosition(pyatspi.XY_SCREEN))
        attribute_dict["{{{:}}}windowcoord".format(_accessibility_ns_map["cp"])] = str(component.getPosition(pyatspi.XY_WINDOW))
        attribute_dict["{{{:}}}parentcoord".format(_accessibility_ns_map["cp"])] = str(component.getPosition(pyatspi.XY_PARENT))
        attribute_dict["{{{:}}}size".format(_accessibility_ns_map["cp"])] = str(component.getSize())
    #  }}} Component # 

    #  Document {{{ # 
    try:
        document: Document = node.queryDocument()
    except NotImplementedError:
        pass
    else:
        attribute_dict["{{{:}}}locale".format(_accessibility_ns_map["doc"])] = document.getLocale()
        attribute_dict["{{{:}}}pagecount".format(_accessibility_ns_map["doc"])] = str(document.getPageCount())
        attribute_dict["{{{:}}}currentpage".format(_accessibility_ns_map["doc"])] = str(document.getCurrentPageNumber())
        for attrbt in document.getAttributes():
            attribute_name: str
            attribute_value: str
            attribute_name, attribute_value = attrbt.split(":", maxsplit=1)
            attribute_dict[ "{{{:}}}{:}"\
                                .format( _accessibility_ns_map["docattr"]
                                       , attribute_name
                                       )
                          ] = attribute_value
    #  }}} Document # 

    #  Text {{{ # 
    try:
        text_obj: ATText = node.queryText()
    except NotImplementedError:
        pass
    else:
        # only text shown on current screen is available
        #attribute_dict["txt:text"] = text_obj.getText(0, text_obj.characterCount)
        text: str = text_obj.getText(0, text_obj.characterCount)
    #  }}} Text # 

    #  Selection {{{ # 
    try:
       node.querySelection()
    except NotImplementedError:
        pass
    else:
        attribute_dict["selection"] = "true"
    #  }}} Selection # 

    #  Value {{{ # 
    try:
        value: ATValue = node.queryValue()
    except NotImplementedError:
        pass
    else:
        attribute_dict["{{{:}}}value".format(_accessibility_ns_map["val"])] = str(value.currentValue)
        attribute_dict["{{{:}}}min".format(_accessibility_ns_map["val"])] = str(value.minimumValue)
        attribute_dict["{{{:}}}max".format(_accessibility_ns_map["val"])] = str(value.maximumValue)
        attribute_dict["{{{:}}}step".format(_accessibility_ns_map["val"])] = str(value.minimumIncrement)
    #  }}} Value # 

    #  Action {{{ # 
    try:
        action: ATAction = node.queryAction()
    except NotImplementedError:
        pass
    else:
        for i in range(action.nActions):
            action_name: str = action.getName(i).replace(" ", "-")
            attribute_dict[ "{{{:}}}{:}_desc"\
                                .format( _accessibility_ns_map["act"]
                                       , action_name
                                       )
                          ] = action.getDescription(i)
            attribute_dict[ "{{{:}}}{:}_kb"\
                                .format( _accessibility_ns_map["act"]
                                       , action_name
                                       )
                          ] = action.getKeyBinding(i)
    #  }}} Action # 

    xml_node = lxml.etree.Element( node.getRoleName().replace(" ", "-")
                                 , attrib=attribute_dict
                                 , nsmap=_accessibility_ns_map
                                 )
    if "text" in locals() and len(text)>0:
        xml_node.text = text
    for ch in node:
        xml_node.append(_create_node(ch))
    return xml_node

def get_thunderbird_writer_at(title: str) -> Optional[_Element]:
    desktop: Accessible = pyatspi.Registry.getDesktop(0)
    found = False
    for app in desktop:
        if app.name=="Thunderbird":
            found = True
            break
    if not found:
        return None

    found = False
    for wnd in app:
        if wnd.name=="Write: {:} - Thunderbird".format(title):
            found = True
            break
    if not found:
        return None

    thunderbird_xml: _Element = _create_node(app)
    return thunderbird_xml

def check_attachment(title: str, name: str) -> bool:
    writer_window: _Element = get_thunderbird_writer_at(title)
    if writer_window is None:
        print("No Thunderbird Instances")
        return False

    bucket_selector = CSSSelector('panel[attr|id="attachmentArea"]>list-box[attr|id="attachmentBucket"]', namespaces=_accessibility_ns_map)
    buckets: List[_Element] = bucket_selector(writer_window)

    if len(buckets)==0:
        button_selector = CSSSelector('panel[attr|id="attachmentArea"]>push-button[name*="Attachment"]', namespaces=_accessibility_ns_map)
        buttons: List[_Element] = button_selector(writer_window)
        if len(buttons)==0:
            print("No attachments attached!")
            return False
        print("Attachments not shown...")
        button: _Element = buttons[0]

        x: int
        y: int
        x, y = eval(button.get("{{{:}}}screencoord".format(_accessibility_ns_map["cp"])))
        w: int
        h: int
        w, h = eval(button.get("{{{:}}}size".format(_accessibility_ns_map["cp"])))

        pyautogui.click(x=(x+w//2), y=(y+h//2))

        time.sleep(.5)
        writer_window = get_thunderbird_writer_at(title)
        buckets = bucket_selector(writer_window)
        while len(buckets)==0:
            time.sleep(1.)
            buckets = bucket_selector(writer_window)

    name = " ".join(os.path.splitext(name))
    item_selector = CSSSelector('list-item[name^="{:}"]'.format(name), namespaces=_accessibility_ns_map)
    return len(item_selector(buckets[0]))>0

if __name__ == "__main__":
    import sys
    subject: str = sys.argv[1]
    attachment: str = sys.argv[2]

    #thunderbird_xml: Optional[_Element] = get_thunderbird_writer_at(subject)
    #if thunderbird_xml is None:
        #print("No Thunderbird Instances")
        #exit(1)

    if check_attachment(subject, attachment):
        print("Attachment added!")
        exit(0)
    print("Attachment not detected!")
    exit(2)
