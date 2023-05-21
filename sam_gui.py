from enum import Enum, auto
from functools import partial
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import os
import yaml
from tkinter.filedialog import askdirectory
import sam_tools

class PromptType(Enum):
    PointP = auto()
    PointN = auto()
    Box = auto()


class GUI():
    def __init__(self, icon_path:str, window_size:tuple, canvas_size:tuple,
                 list_size:tuple, img_size:tuple, marker_size:tuple,
                 label_font:tuple, button_font:tuple, palette:dict,
                 load_dir:str, save_dir:str, sam_info:dict) -> None:
        self.window_size = window_size
        self.canvas_size = canvas_size
        self.list_size = list_size
        self.img_size = img_size
        self.point_size = marker_size["point"]
        self.box_width = marker_size["box"]
        self.point_width = marker_size["point_width"]
        self.palette = palette
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.sam_info = sam_info
        self.root = tk.Tk()
        self.root.title("Powered by Tkinter")
        self.root.iconphoto(True,tk.PhotoImage(file=icon_path))
        MF = tk.Frame(self.root)
        self.strvar = tk.StringVar(value="Status:")
        self.statusbar = tk.Label(self.root,textvariable=self.strvar,relief=tk.SUNKEN, anchor="w",foreground="blue",font=label_font)
        self.statusbar.pack(side=tk.BOTTOM,fill=tk.X)
        MF.pack(side=tk.TOP)
        MF1 = tk.Frame(MF)
        MF2 = tk.Frame(MF)
        MF1.pack(side=tk.LEFT)
        MF2.pack(side=tk.RIGHT)
        F1_heading = tk.Label(MF1, text="Canvas", font=label_font)
        F2_heading = tk.Label(MF2, text="Panel", font=label_font)
        F1_heading.pack(side=tk.TOP)
        F2_heading.pack(side=tk.TOP)
        F1 = tk.Frame(MF1,borderwidth=2)
        F2 = tk.Frame(MF2,borderwidth=2)
        F1.pack(side=tk.BOTTOM)
        F2.pack(side=tk.BOTTOM)
        F11 = tk.Frame(F1)
        F11.pack(side=tk.TOP)
        F12 = tk.Frame(F1,background="#AA43BB")
        F12.pack(side=tk.BOTTOM,expand=True)
        tk.Label(F11, text="Mask Selection", font=label_font).pack(side=tk.LEFT)
        self.scale = tk.Scale(F11, cursor="circle",length=150,orient=tk.HORIZONTAL, showvalue=1, from_=1,to=3,resolution=1)
        self.scale.pack(side=tk.RIGHT)
        
        self.canvas_hbar = tk.Scrollbar(F12, orient=tk.HORIZONTAL)
        self.canvas_vbar = tk.Scrollbar(F12, orient=tk.VERTICAL)
        self.canvas = tk.Canvas(F12, width=self.canvas_size[1], height=self.canvas_size[0],
                                cursor="tcross",bg=palette["canvas"],
                                scrollregion=[0,0,img_size[1],img_size[0]],
                                xscrollcommand=self.canvas_hbar.set,yscrollcommand=self.canvas_vbar.set)
        self.canvas_hbar.config(command=self.canvas.xview)
        self.canvas_vbar.config(command=self.canvas.yview)
        
        self.canvas_vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(fill=tk.BOTH)
        self.canvas.bind("<Button-1>",self.add_marker)
        
        F21 = tk.Frame(F2)
        F22 = tk.Frame(F2)
        F23 = tk.Frame(F2)
        F24 = tk.Frame(F2)
        F21.pack(side=tk.TOP)
        self.prompt_label = tk.Label(F2, text="Prompt Type", font=label_font)
        self.prompt_label.pack(side=tk.TOP)
        F22.pack(side=tk.TOP)
        F23.pack(side=tk.TOP)
        tk.Label(F2, text="Markers", font=label_font).pack(side=tk.TOP)
        F24.pack(side=tk.TOP)
        
        self.load_button = tk.Button(F21,text="Load Dir",font=button_font,command=self.set_load_dir)
        self.save_buttion = tk.Button(F21, text="Save Dir",font=button_font, command=self.set_save_dir)
        self.load_model_button = tk.Button(F21, text="Load Model",font=button_font, command=self.load_model)
        self.load_button.pack(side=tk.LEFT,padx=5)
        self.save_buttion.pack(side=tk.LEFT,padx=5)
        self.load_model_button.pack(side=tk.LEFT,padx=5)
        
        self.pos_point_buttion = tk.Button(F22, text="Point (P)",font=button_font, command=partial(self.set_prompt_type,PromptType.PointP))
        self.neg_point_buttion = tk.Button(F22, text="Point (N)",font=button_font, command=partial(self.set_prompt_type,PromptType.PointN))
        self.box_button = tk.Button(F22, text="Box",font=button_font, command=partial(self.set_prompt_type,PromptType.Box))
        self.pos_point_buttion.pack(side=tk.LEFT,padx=3)
        self.neg_point_buttion.pack(side=tk.LEFT,padx=3)
        self.box_button.pack(side=tk.RIGHT,padx=3)
        
        F231 = tk.Frame(F23)
        F232 = tk.Frame(F23)
        F231.pack(side=tk.LEFT)
        F232.pack(side=tk.RIGHT)
        imglbox_vbar = tk.Scrollbar(F231, orient=tk.VERTICAL)
        savelbox_vbar = tk.Scrollbar(F232, orient=tk.VERTICAL)
        self.imglbox = tk.Listbox(F231, yscrollcommand=imglbox_vbar.set, selectmode=tk.SINGLE,
                                  width=list_size[0],height=list_size[1],font=button_font)
        self.savelbox = tk.Listbox(F232, yscrollcommand=savelbox_vbar.set, selectmode=tk.SINGLE,
                                  width=list_size[0],height=list_size[1],font=button_font)
        imglbox_vbar.config(command=self.imglbox.yview)
        savelbox_vbar.config(command=self.savelbox.yview)
        
        imglbox_vbar.pack(side=tk.RIGHT,fill=tk.Y)
        self.imglbox.pack(fill=tk.BOTH)
        savelbox_vbar.pack(side=tk.RIGHT,fill=tk.Y)
        self.savelbox.pack(fill=tk.BOTH)

        self.imglbox.bind("<<ListboxSelect>>",self.imglbox_callback)
        
        self.prompt_type = None
        self.imglist = list(sorted(os.listdir(self.load_dir)))
        self.savelist = list(sorted(os.listdir(self.save_dir)))
        
        F241 = tk.Frame(F24)
        F242 = tk.Frame(F24)
        F241.pack(side=tk.LEFT)
        F242.pack(side=tk.RIGHT)
        taglbox_vbar = tk.Scrollbar(F241, orient=tk.VERTICAL)
        self.taglbox = tk.Listbox(F241, yscrollcommand=taglbox_vbar.set, selectmode=tk.SINGLE,
                                  width=list_size[0],height=list_size[1],font=button_font)
        taglbox_vbar.config(command=self.taglbox.yview)
        self.taglbox.pack(side=tk.LEFT, fill=tk.BOTH)
        taglbox_vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.taglbox.bind("<<ListboxSelect>>",self.highlight_tag)
        
        self.predict_button = tk.Button(F242, text="Predict",font=button_font,command=self.predict_mask)
        self.delete_tag_button = tk.Button(F242, text="Delete",font=button_font,command=self.delete_select_tag)
        self.delete_all_tags_button = tk.Button(F242, text="Delete All",font=button_font,command=self.delete_all_tags)
        self.flush_button = tk.Button(F242, text="Flush",font=button_font,command=self.flush_canvas)
        self.predict_button.pack(side=tk.TOP,pady=3)
        self.delete_tag_button.pack(side=tk.TOP,pady=3)
        self.delete_all_tags_button.pack(side=tk.TOP,pady=3)
        self.flush_button.pack(side=tk.TOP,pady=3)
        # Temporary Variables
        self.imgetk = None   
        self.masktk = None
        self.curr_img_file = None
        self.input_img_file = None     
        
        self.markers = []
        self.canvas_tags = []
        self.canvas_tmp_tags = []
        # idx for marker tags
        self.PPcnt = 0
        self.NPcnt = 0
        self.Boxcnt = 0
        # sam
        self.sam_model = None
        self.masks = None  # [3, H, W]
        self.scores = np.zeros(3)
        self.prev_logits = None  # [1, H, W]
        
    def set_prompt_type(self, prompt_type:Enum):
        if(prompt_type == PromptType.PointP):
            self.prompt_label.config(text="Positive Point",foreground=self.palette["Point-P"])
            self.strvar.set("PromptType set to Point-P")
        elif(prompt_type == PromptType.PointN):
            self.prompt_label.config(text="Negative Point",foreground=self.palette["Point-N"])
            self.strvar.set("PromptType set to Point-N")
        elif(prompt_type == PromptType.Box):
            self.prompt_label.config(text="Box",foreground=self.palette["Box"])
            self.strvar.set("PromptType set to Box")
        else:
            prompt_type = None
            self.strvar.set("Empty Prompt")
        self.prompt_type = prompt_type
    
    @staticmethod
    def marker_to_prompts(markers:list):
        input_points = []
        point_labels = []
        input_boxes = []
        for marker in markers:
            if(marker['type'] == PromptType.PointP.value):
                input_points.append(marker["coord"])
                point_labels.append(1)
            elif(marker['type'] == PromptType.PointN.value):
                input_points.append(marker["coord"])
                point_labels.append(0)
            elif(marker['type'] == PromptType.Box.value):
                input_boxes.append(marker["coord"])
        if len(input_points) == 0:
            input_points = None
            point_labels = None
        else:
            input_points = np.array(input_points)
            point_labels = np.array(point_labels)
        if len(input_boxes) == 0:
            input_boxes = None
        else:
            input_boxes = np.array(input_boxes)
        return input_points, point_labels, input_boxes
    
    def load_model(self):
        self.strvar.set("Loading SAM Model from %s..."%(self.sam_info["checkpoint"]))
        self.root.update()
        self.sam_model = sam_tools.get_model(self.sam_info["model"], self.sam_info["checkpoint"], self.sam_info["device"])
        self.strvar.set("SAM Model loaded.")
    
    def run(self) -> None:
        self.flush_imglbox()
        self.flush_savelbox()
        self.strvar.set("Click Load Model Button First.")
        self.root.mainloop()
        
        
    def set_load_dir(self):
        tmp:str = askdirectory(initialdir=self.load_dir,title="Directory of Loading Images")
        if tmp and tmp != self.load_dir:
            self.load_dir = tmp
            self.imglist = list(sorted(os.listdir(self.load_dir)))
            self.flush_imglbox()
            self.strvar.set("Load Dir changed to %s"%self.load_dir)
            
    def set_save_dir(self):
        tmp:str = askdirectory(initialdir=self.save_dir,title="Directory of Saving Images")
        if tmp and tmp != self.save_dir:
            self.save_dir = tmp
            self.savelist = list(sorted(os.listdir(self.save_dir)))
            self.flush_savelbox()
            self.strvar.set("Save Dir changed to %s"%self.save_dir)
    
    def flush_imglbox(self):
        self.imglbox.delete(0,tk.END)
        for imgfilename in self.imglist:
            self.imglbox.insert(tk.END, imgfilename)
        
    def flush_savelbox(self):
        self.savelbox.delete(0,tk.END)
        for savefilename in self.savelist:
            self.savelbox.insert(tk.END, savefilename)
        
    def imglbox_callback(self,event:tk.Event=None):
        index = self.imglbox.curselection()
        if len(index) == 0:
            if event.widget == self.imglbox:
                self.strvar.set("No Image Selected for loading.")
            return
        idx = index[0]
        self.load_img_to_canvas(os.path.join(self.load_dir, self.imglist[idx]))
        
    def load_img_to_canvas(self, imgfile):
        image = Image.open(imgfile)
        if(image.size[0] != self.img_size[1] or image.size[1] != self.img_size[0]):
            image.resize([self.img_size[1], self.img_size[0]],resample=Image.Resampling.BILINEAR)
        self.strvar.set("%s loaded. (H: %d, W: %d)"%(imgfile, image.height, image.width))
        self.imagetk = ImageTk.PhotoImage(image)
        self.canvas.create_image(0,0,anchor="nw",image=self.imagetk,tag="image")
        self.curr_img_file = imgfile
        
    def flush_canvas(self):
        """redraw the canvas with the current image and all cached markers (excluding temporary markers)
        """
        self.canvas.delete(tk.ALL)
        self.taglbox.delete(0,tk.END)
        self.canvas_tags.clear()
        self.canvas_tmp_tags.clear()
        self.canvas.create_image(0,0,anchor="nw",image=self.imagetk,tag="image")
        for marker in self.markers:
            if not isinstance(marker, dict):
                continue
            self.add_marker_to_canvas(marker)
    
    def add_marker_to_canvas(self, marker:dict):
        if(marker['type'] == PromptType.PointP.value):
            tag = "PP-%03d"%(self.PPcnt)
            self.PPcnt += 1
            self.canvas.create_oval(marker['coord'][0]-self.point_size,marker['coord'][1]-self.point_size,
                                    marker['coord'][0]+self.point_size,marker['coord'][1]+self.point_size,
                                    fill=self.palette["Point-P"],tags=tag,width=self.point_width)
            self.canvas_tags.append(tag)
            self.taglbox.insert(tk.END, tag)
            
        elif(marker['type'] == PromptType.PointN.value):
            tag = "NP-%03d"%(self.NPcnt)
            self.NPcnt += 1
            self.canvas.create_oval(marker['coord'][0]-self.point_size,marker['coord'][1]-self.point_size,
                                    marker['coord'][0]+self.point_size,marker['coord'][1]+self.point_size,
                                    fill=self.palette["Point-N"],tags=tag,width=self.point_width)
            self.canvas_tags.append(tag)
            self.taglbox.insert(tk.END, tag)
            
        elif(marker['type'] == PromptType.Box.value):
            tag = "Box-%03d"%(self.Boxcnt)
            self.Boxcnt += 1
            self.canvas.create_rectangle(marker['coord'][0],marker['coord'][1],
                                            marker['coord'][2],marker['coord'][3],
                                    fill="",bg=self.palette["Box"],tags=tag,width=self.box_width)
            self.canvas_tags.append(tag)
            self.taglbox.insert(tk.END, tag)
        else:
            self.strvar.set("Invalid Marker.")
    
    
    def delete_select_tag(self):
        index = self.taglbox.curselection()
        if len(index) == 0:
            return
        idx = index[0]
        tag = self.canvas_tags[idx]
        self.canvas.delete(tag)
        self.canvas_tags.pop(idx)
        self.markers.pop(idx)
        self.taglbox.delete(idx)
        for tag in self.canvas_tmp_tags:
            self.canvas.delete(tag)
        self.canvas_tmp_tags.clear()
        self.strvar.set("Tag %s Deleted."%tag)
        
    def delete_all_tags(self):
        for tag in self.canvas_tags:
            self.canvas.delete(tag)
        self.taglbox.delete(0,tk.END)
        for tag in self.canvas_tmp_tags:
            self.canvas.delete(tag)
        self.markers.clear()
        self.PPcnt = 0
        self.NPcnt = 0
        self.Boxcnt = 0
        self.canvas_tmp_tags.clear()
        self.strvar.set("%d tags Deleted."%len(self.canvas_tags))
        self.canvas_tags.clear()
        
    def highlight_tag(self, event:tk.Event=None):
        index = self.taglbox.curselection()
        if len(index) == 0:
            return
        idx = index[0]
        for tag in self.canvas_tmp_tags:
            self.canvas.delete(tag)
        self.canvas_tmp_tags.clear()
        marker = self.markers[idx]
        tag = "tmp"
        if(marker["type"] == PromptType.PointP.value or marker["type"]==PromptType.PointN.value):
            if marker["type"] == PromptType.PointP.value:
                outlinecolor = self.palette["Point-P"]
            elif marker["type"] == PromptType.PointN.value:
                outlinecolor = self.palette["Point-N"]
            self.canvas.create_oval(marker['coord'][0]-self.point_size,marker['coord'][1]-self.point_size,
                                    marker['coord'][0]+self.point_size,marker['coord'][1]+self.point_size,
                                    fill=self.palette["Highlight"],tags=tag,width=self.point_width,outline=outlinecolor)
            self.canvas_tmp_tags.append(tag)
            see_x = (marker['coord'][0]-self.canvas_size[1])/(self.img_size[1] - self.canvas_size[1])
            see_y = (marker['coord'][1]-self.canvas_size[0])/(self.img_size[0] - self.canvas_size[0])
            see_x = max(see_x,0)
            see_y = max(see_y,0)
            self.canvas.xview_moveto(see_x)
            self.canvas.yview_moveto(see_y)
        elif marker['type'] == PromptType.Box.value:
            self.canvas.create_rectangle(marker['coord'][0],marker['coord'][1],
                                        marker['coord'][2],marker['coord'][3],
                                    fill="",bg=self.palette["Highlight"],tags=tag,width=self.box_width)
            self.canvas_tmp_tags.append(tag)
            see_x = (marker['coord'][0]-self.canvas_size[1])/(self.img_size[1] - self.canvas_size[1])
            see_y = (marker['coord'][1]-self.canvas_size[0])/(self.img_size[0] - self.canvas_size[0])
            see_x = max(see_x,0)
            see_y = max(see_y,0)
            self.canvas.xview_moveto(see_x)
            self.canvas.yview_moveto(see_y)
            
    def highlight_mask(self, mask:np.ndarray):
        """Highlight selected mask

        Args:
            mask (np.ndarray): H x W
        """
        color = np.array(self.palette["mask"])  # (4,)
        mask_arr = np.array(mask[...,None] * color[None,None,:],dtype=np.uint8)
        mask_image = Image.fromarray(mask_arr)  # (H,W,1) * (1,1,4) -> (H,W,4)
        self.masktk = ImageTk.PhotoImage(mask_image)
        self.canvas.delete("mask")
        self.canvas.create_image(0,0,anchor="nw", image=self.masktk,tag="mask")
        
    def predict_mask(self):
        self.strvar.set("Predicting by SAM %s using %s"%(self.sam_info["model"], self.sam_info["device"]))
        self.root.update()
        if self.input_img_file != self.curr_img_file:
            self.input_img_file = self.curr_img_file
            image = Image.open(self.input_img_file)
            sam_tools.set_image(self.sam_model, np.array(image)[...,:3], "RGB")  # retrive first 3 channels of image (ignore alpha channel if has)
        input_points, point_labels, input_boxes = self.marker_to_prompts(self.markers)
        self.masks, self.scores, logits = sam_tools.mask_predict(self.sam_model,input_points, point_labels, input_boxes,self.prev_logits,True)
        self.strvar.set("Predict Completed.")
        max_score_id = np.argmax(self.scores)
        self.scale.set(max_score_id+1)
        self.prev_logits = logits[[max_score_id],...]  # [1, H, W]
        self.highlight_mask(self.masks[max_score_id,...])
        
    def add_marker(self, event:tk.Event):
        mx = self.canvas.canvasx(event.x)
        my = self.canvas.canvasy(event.y)
        if self.prompt_type is None:
            self.strvar.set("Choose a Prompt Type First")
        elif self.prompt_type == PromptType.PointP:
            self.add_PP(mx, my)
            self.strvar.set("Positive Point Added (%d, %d)"%(mx, my))
        elif self.prompt_type == PromptType.PointN:
            self.add_NP(mx, my)
            self.strvar.set("Negative Point Added (%d, %d)"%(mx, my))
    
    def add_PP(self, mx:int, my:int):
        # mouse position
        marker={"type":PromptType.PointP.value, "coord":[mx,my]}
        self.markers.append(marker)
        self.add_marker_to_canvas(marker)
        
    def add_NP(self, mx:int, my:int):
        # mouse position
        marker={"type":PromptType.PointN.value, "coord":[mx,my]}
        self.markers.append(marker)
        self.add_marker_to_canvas(marker)
        
    

if __name__ == "__main__":
    config = yaml.load(open("config.yml",'r'),Loader=yaml.SafeLoader)
    gui = GUI(config["gui"]["icon"], config["gui"]["window_size"], config["gui"]["canvas_size"],
              config["gui"]["list_size"], config["io"]["image_size"], config["gui"]["marker_size"],
              config["gui"]["label_font"], config["gui"]["button_font"], config["gui"]["palette"],
              config["io"]["image_dir"], config["io"]["mask_dir"], config["sam"])
    gui.run()

# fig = Figure()
# canvas = FigureCanvas(fig)
# ax = fig.gca()

# ax.text(0.0,0.0,"Test", fontsize=45)
# ax.axis('off')

# canvas.draw()       # draw the canvas, cache the renderer

# image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
# image = image_flat.reshape(*fig.canvas.get_width_height(), 3)  # (H, W, 3)
# img = Image.fromarray(image)
# img.show()