from enum import Enum, auto
from functools import partial
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import os
import yaml
from tkinter.filedialog import askdirectory, askopenfilename
import sam_tools

class PromptType(Enum):
    PointP = auto()
    PointN = auto()
    Box = auto()

class BoxState(Enum):
    Hanging = auto()
    Release = auto()

class GUI():
    def __init__(self, icon_path:str, logit_size:tuple, canvas_size:tuple,
                 file_list_size:tuple, marker_list_size:tuple, img_size:tuple, marker_size:tuple,
                 label_font:tuple, button_font:tuple, item_font, palette:dict,
                 load_dir:str, save_dir:str, sam_info:dict) -> None:
        # H, W
        self.logit_size = logit_size  # has not been used yet
        self.canvas_size = canvas_size
        self.img_size = img_size 
        self.raw_img_size = img_size  # will change if the input_size != img_size
        self.point_size = marker_size["point"]
        self.box_width = marker_size["box"]
        self.point_width = marker_size["point_width"]
        self.palette = palette
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.sam_info = sam_info
        self.root = tk.Tk()
        self.root.title("SAM GUI")
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
        F1_heading = tk.Label(MF1, text="Canvas", font=label_font,relief=tk.SUNKEN)
        F2_heading = tk.Label(MF2, text="Panel", font=label_font, relief=tk.SUNKEN)
        F1_heading.pack(side=tk.TOP,pady=10)
        F2_heading.pack(side=tk.TOP,pady=10)
        F1 = tk.Frame(MF1,borderwidth=2)
        F2 = tk.Frame(MF2,borderwidth=2)
        F1.pack(side=tk.BOTTOM)
        F2.pack(side=tk.BOTTOM)
        F11 = tk.Frame(F1)
        F11.pack(side=tk.TOP)
        F12 = tk.Frame(F1,background="#AA43BB")
        F12.pack(side=tk.BOTTOM,expand=True)
        tk.Label(F11, text="Mask Selection", font=label_font).pack(side=tk.LEFT,padx=5)
        self.scale = tk.Scale(F11, cursor="circle",length=150,orient=tk.HORIZONTAL, showvalue=1, from_=1,to=3,resolution=1,command=self.choose_mask)
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
        self.canvas.bind("<Motion>",self.adding_rectangle)
        
        F21 = tk.Frame(F2)  # IO setting
        F22 = tk.Frame(F2)  # Prompts
        F23 = tk.Frame(F2)  # Marker Operations
        F24 = tk.Frame(F2)  # img list and save list
        F25 = tk.Frame(F2)  # File operations
        
        F21.pack(side=tk.TOP)
        self.prompt_label = tk.Label(F2, text="Prompt Type", font=label_font)
        self.prompt_label.pack(side=tk.TOP,pady=5)
        F22.pack(side=tk.TOP)
        tk.Label(F2, text="Marker", font=label_font).pack(side=tk.TOP,pady=5)
        F23.pack(side=tk.TOP)
        tk.Label(F2, text="File View", font=label_font).pack(side=tk.TOP,pady=5)
        F24.pack(side=tk.TOP)
        F25.pack(side=tk.TOP)
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
        LF23 = tk.Frame(F23)
        RF23 = tk.Frame(F23)
        LF23.pack(side=tk.LEFT)
        RF23.pack(side=tk.RIGHT)
        F231 = tk.Frame(LF23)
        F232 = tk.Frame(RF23)
        F233 = tk.Frame(RF23)
        F231.pack(side=tk.TOP,pady=5)
        F232.pack(side=tk.TOP,pady=5)
        F233.pack(side=tk.TOP,pady=5)
        taglbox_vbar = tk.Scrollbar(F231, orient=tk.VERTICAL)
        self.taglbox = tk.Listbox(F231, yscrollcommand=taglbox_vbar.set, selectmode=tk.SINGLE,
                                  width=marker_list_size[0],height=marker_list_size[1],font=button_font)
        taglbox_vbar.config(command=self.taglbox.yview)
        self.taglbox.pack(side=tk.LEFT, fill=tk.BOTH)
        taglbox_vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.taglbox.bind("<<ListboxSelect>>",self.highlight_tag)
        
        self.predict_button = tk.Button(F232, text="Predict",font=button_font,command=self.predict_mask)
        self.delete_tag_button = tk.Button(F232, text="Delete",font=button_font,command=self.delete_select_tag)
        self.delete_all_tags_button = tk.Button(F232, text="Delete All",font=button_font,command=self.delete_all_tags)
        self.flush_button = tk.Button(F233, text="Flush Canvas",font=button_font,command=self.flush_all)
        self.clear_cache_button = tk.Button(F233, text="Clear Cache",font=button_font,command=self.clear_model_cache)
        self.predict_button.pack(side=tk.LEFT,pady=3,padx=3)
        self.delete_tag_button.pack(side=tk.LEFT,pady=3,padx=3)
        self.delete_all_tags_button.pack(side=tk.LEFT,pady=3,padx=3)
        self.flush_button.pack(side=tk.LEFT,pady=3,padx=3)
        self.clear_cache_button.pack(side=tk.LEFT,pady=3,padx=3)
        
        F241 = tk.Frame(F24)
        F242 = tk.Frame(F24)
        F243 = tk.Frame(F24)
        F241.pack(side=tk.LEFT)
        F242.pack(side=tk.LEFT)
        F243.pack(side=tk.LEFT)
        imglbox_vbar = tk.Scrollbar(F241, orient=tk.VERTICAL)
        savelbox_vbar = tk.Scrollbar(F242, orient=tk.VERTICAL)
        self.imglbox = tk.Listbox(F241, yscrollcommand=imglbox_vbar.set, selectmode=tk.SINGLE,
                                  width=file_list_size[0],height=file_list_size[1],font=button_font)
        self.savelbox = tk.Listbox(F242, yscrollcommand=savelbox_vbar.set, selectmode=tk.SINGLE,
                                  width=file_list_size[0],height=file_list_size[1],font=button_font)
        imglbox_vbar.config(command=self.imglbox.yview)
        savelbox_vbar.config(command=self.savelbox.yview)
        
        imglbox_vbar.pack(side=tk.RIGHT,fill=tk.Y)
        self.imglbox.pack(fill=tk.BOTH)
        savelbox_vbar.pack(side=tk.RIGHT,fill=tk.Y)
        self.savelbox.pack(fill=tk.BOTH)
        self.imglbox.bind("<<ListboxSelect>>",self.imglbox_callback)
        tk.Label(F243,text="Options",font=button_font).pack(side=tk.TOP,pady=3)  # smaller
        self.auto_save_var = tk.IntVar(value=1)
        self.auto_save_check = tk.Checkbutton(F243, text="auto save", font=item_font, variable=self.auto_save_var)
        self.auto_save_check.pack(side=tk.TOP, padx=5,pady=3)
        self.auto_loadmask_var = tk.IntVar(value=1)
        self.auto_loadmask_check = tk.Checkbutton(F243, text="check mode", font=item_font, variable=self.auto_loadmask_var)
        self.auto_loadmask_check.pack(side=tk.TOP, padx=5, pady=3)
        self.overwrite_config_var = tk.IntVar(value=1)
        self.overwrite_config_check = tk.Checkbutton(F243, text="overwrite config", font=item_font, variable=self.overwrite_config_var)
        self.overwrite_config_check.pack(side=tk.TOP, padx=5, pady=3)
        
        F251 = tk.Frame(F25)
        F252 = tk.Frame(F25)
        F253 = tk.Frame(F25)
        F251.pack(side=tk.LEFT)
        F252.pack(side=tk.LEFT)
        F253.pack(side=tk.LEFT)
        self.save_mask_button = tk.Button(F251,text="Save Mask",font=button_font,command=self.save_mask)
        self.load_mask_button = tk.Button(F251,text="Load Mask",font=button_font,command=self.load_mask)
        self.next_image_button = tk.Button(F252, text="Next Image",font=button_font,command=self.next_image)
        self.last_image_button = tk.Button(F252, text="Last Image",font=button_font,command=self.last_image)
        self.save_mask_button.pack(side=tk.TOP,padx = 5, pady=3)
        self.load_mask_button.pack(side=tk.TOP,padx = 5, pady=3)
        self.next_image_button.pack(side=tk.TOP,padx = 5, pady=3)
        self.last_image_button.pack(side=tk.TOP,padx = 5, pady=3)
        tk.Label(F253,text="Model",font=button_font).pack(side=tk.TOP,pady=3)  # smaller
        self.model_type_var = tk.IntVar()
        self.model_types = ['vit_b','vit_l','vit_h']
        self.model_type_var.set(self.model_types.index(self.sam_info["model"]))
        self.vit_b_radio = tk.Radiobutton(F253, text="vit_b", variable=self.model_type_var,value=0,command=self.set_model_type)
        self.vit_l_radio = tk.Radiobutton(F253, text="vit_l", variable=self.model_type_var,value=1,command=self.set_model_type)
        self.vit_h_radio = tk.Radiobutton(F253, text="vit_h", variable=self.model_type_var,value=2,command=self.set_model_type)
        self.vit_b_radio.pack(side=tk.TOP)
        self.vit_l_radio.pack(side=tk.TOP)
        self.vit_h_radio.pack(side=tk.TOP)
        # Hot Key
        self.root.bind("<Control-p>",self.predict_mask)
        self.root.bind("<Control-P>",self.predict_mask)
        self.root.bind("<Control-m>",self.see_relevant_maskfile)
        self.root.bind("<Control-M>",self.see_relevant_maskfile)
        self.root.bind("<Control-l>",self.load_mask)
        self.root.bind("<Control-L>",self.load_mask)
        self.root.bind("<Control-s>",self.save_mask)
        self.root.bind("<Control-S>",self.save_mask)
        self.root.bind("<Control-Down>",self.next_image)
        self.root.bind("<Control-Up>",self.last_image)
        self.root.bind("<Escape>",self.cancel_adding_rectangle)
        
        # Nonlocal Variables
        if os.path.exists(self.load_dir):
            self.imglist = list(sorted(os.listdir(self.load_dir)))
        else:
            self.imglist = []
        if os.path.exists(self.save_dir):
            self.savelist = list(sorted(os.listdir(self.save_dir)))
        else:
            self.savelist = []
        self.imgetk = None   
        self.masktk = None
        self.curr_img_idx = 0
        self.curr_img_file = None
        self.input_img_file = None     
        self.input_img_arr = None
        self.prompt_type = None  # PromptType Enum
        self.box_state = BoxState.Release  # BoxState Enum
        self.box_buff = None
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
        self.logits = None  # [1, H, W]
        self.prev_select = 0  # 0~3
        self.coord_tran = None
        self.scores = np.zeros(3)
    
    def set_prompt_type(self, prompt_type:Enum):
        if(prompt_type == PromptType.PointP):
            self.prompt_label.config(text="Positive Point",foreground=self.palette["Point-P"])
            self.strvar.set("PromptType set to Point-P")
            self.cancel_adding_rectangle()
        elif(prompt_type == PromptType.PointN):
            self.prompt_label.config(text="Negative Point",foreground=self.palette["Point-N"])
            self.strvar.set("PromptType set to Point-N")
            self.cancel_adding_rectangle()
        elif(prompt_type == PromptType.Box):
            self.prompt_label.config(text="Box",foreground=self.palette["Box"])
            self.strvar.set("PromptType set to Box")
            self.cancel_adding_rectangle()
        else:
            prompt_type = None
            self.strvar.set("Empty Prompt")
        self.prompt_type = prompt_type
        self.clear_tmp_markers()
    
    def set_model_type(self):
        val = self.model_type_var.get()
        if self.sam_info["model"] != self.model_types[val]:
            self.sam_info["model"] = self.model_types[val]
            self.strvar.set("Model Type Changed to %s. (Need to Reload the model)"%self.model_types[val])
    
    def marker_to_prompts(self, markers:list):
        input_points = []
        point_labels = []
        input_boxes = []
        for marker in markers:
            coord = self.coord_tran(marker["coord"])
            if(marker['type'] == PromptType.PointP.value):
                input_points.append(coord)
                point_labels.append(1)
            elif(marker['type'] == PromptType.PointN.value):
                input_points.append(coord)
                point_labels.append(0)
            elif(marker['type'] == PromptType.Box.value):
                input_boxes.append(coord)
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
        model_path = askopenfilename(defaultextension=".pth",
                                 initialdir=os.path.dirname(self.sam_info["checkpoint"]),
                                 initialfile=os.path.basename(self.sam_info["checkpoint"]),
                                 title="Load Checkpoint")
        if not (model_path and os.path.exists(model_path)):
            self.strvar.set("Did not load SAM Model.")
            return
        self.sam_info["checkpoint"] = model_path
        self.strvar.set("Loading SAM Model from %s..."%(model_path))
        self.root.update()
        real_model_size = os.stat(model_path).st_size / 1024 / 1024 / 1024  # (GB)
        ref_model_size = self.sam_info["model_size"][self.sam_info["model"]]
        if abs(real_model_size - ref_model_size) / ref_model_size > 0.2:
            self.strvar.set("SAM model not loaded - (%0.2f GB != %s model: %0.2f GB), please Check!"%(real_model_size, self.sam_info["model"], ref_model_size))
            return
        self.sam_model = sam_tools.get_model(self.sam_info["model"], model_path, self.sam_info["device"])
        self.strvar.set("%s SAM Model loaded (%.2f GB) to %s."%(self.sam_info["model"], real_model_size, self.sam_info["device"]))

    def save_mask(self, event:tk.Event=None):
        if self.masks is None:
            messagebox.showerror(title="IO Error",message="No mask to save!")
            return
        mask = np.array(self.masks[self.prev_select,...],dtype=np.uint8) # (H,W) (0,1)
        mask_image = Image.fromarray(mask)
        mask_basename = os.path.splitext(os.path.basename(self.curr_img_file))[0] + ".jpg"
        mask_full_path = os.path.join(self.save_dir, mask_basename)
        if mask_basename not in self.savelist:
            self.savelist.append(mask_basename)
            self.savelbox.insert(tk.END, mask_basename)
            self.strvar.set("Selected Mask %d has been saved to %s"%(self.prev_select+1, mask_full_path))
        else:
            self.strvar.set("Selected Mask %d has been overwritted to %s"%(self.prev_select+1, mask_full_path))
        mask_image.save(mask_full_path)
        
    def load_mask(self, event:tk.Event=None):
        if self.input_img_arr is None:
            messagebox.showwarning(title="Image Empty",message="Image is Empty, load it first!")
            return
        index = self.savelbox.curselection()
        if len(index) == 0:
            messagebox.showerror(title="IO Error",message="No mask to load!")
            return
        idx = index[0]
        mask_full_path = os.path.join(self.save_dir, self.savelist[idx])
        mask_image = Image.open(mask_full_path)
        
        self.masks = np.repeat(
            np.array(mask_image.resize(sam_tools.reverse_size(self.raw_img_size),Image.Resampling.NEAREST))[None,...],
            repeats=3,
            axis=0)
        self.logits = np.repeat(
            np.array(mask_image.resize(sam_tools.reverse_size(self.logit_size), Image.Resampling.NEAREST))[None,...],
            repeats=3,
            axis=0
        )
        self.scores = np.zeros(3)
        self.highlight_mask(self.masks[0,...])
        self.strvar.set("mask loaded from %s"%(mask_full_path))
    
    def next_image(self, event:tk.Event=None):
        if self.auto_save_var.get() == 1 and self.masks is not None:
            self.save_mask()
        if self.curr_img_idx < len(self.imglist) - 1:
            self.curr_img_idx += 1
            self.imglbox.select_clear(0,tk.END)
            self.imglbox.select_set(self.curr_img_idx)
            self.imglbox_callback()
        else:
            self.strvar.set("This is the last image.")
            
    def last_image(self, event:tk.Event=None):
        if self.auto_save_var.get() == 1 and self.masks is not None:
            self.save_mask()
        if self.curr_img_idx >= 1:
            self.curr_img_idx -= 1
            self.imglbox.select_clear(0,tk.END)
            self.imglbox.select_set(self.curr_img_idx)
            self.imglbox_callback()
        else:
            self.strvar.set("This is the first image.")
    
    def run(self) -> None:
        self.flush_imglbox()
        self.flush_savelbox()
        self.strvar.set("Click 'Load Model' Button First.")
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
    
    def flush_all(self):
        self.flush_canvas()
        self.flush_imglbox()
        self.flush_savelbox()
    
    def flush_imglbox(self):
        if os.path.exists(self.load_dir):
            self.imglist = list(sorted(os.listdir(self.load_dir)))
        else:
            self.imglist = []
        self.imglbox.delete(0,tk.END)
        for imgfilename in self.imglist:
            self.imglbox.insert(tk.END, imgfilename)
        
    def flush_savelbox(self):
        if os.path.exists(self.save_dir):
            self.savelist = list(sorted(os.listdir(self.save_dir)))
        else:
            self.savelist = []
        self.savelbox.delete(0,tk.END)
        for savefilename in self.savelist:
            self.savelbox.insert(tk.END, savefilename)
        
    def imglbox_callback(self,event:tk.Event=None):
        index = self.imglbox.curselection()
        if len(index) == 0:
            return
        idx = index[0]
        self.curr_img_idx = idx
        self.load_img_to_canvas(os.path.join(self.load_dir, self.imglist[idx]))
        if self.auto_loadmask_var.get() > 0:
            res = self.see_relevant_maskfile()
            if res:
                self.load_mask()
            
    
    def see_relevant_maskfile(self, event:tk.Event=None):
        img_name = os.path.splitext(self.imglist[self.curr_img_idx])[0]
        mask_names = [os.path.splitext(name)[0] for name in self.savelist]
        if img_name in mask_names:
            mask_idx = mask_names.index(img_name)
            self.savelbox.see(mask_idx)
            self.savelbox.select_clear(0,tk.END)
            self.savelbox.select_set(mask_idx)
            return True
        else:
            self.strvar.set("Cannot Find %s in Save List. (Maybe Flush button can fix it)"%img_name)
            return False
    
    def load_img_to_canvas(self, imgfile):
        image = Image.open(imgfile)
        self.input_img_arr:np.ndarray = np.array(image)[...,:3]  # H, W, 3
        self.raw_img_size = self.input_img_arr.shape[:2]
        image = image.resize(sam_tools.reverse_size(self.img_size),resample=Image.Resampling.BILINEAR)
        self.coord_tran = partial(sam_tools.coord_tran, kx=self.raw_img_size[0]/self.img_size[0], ky=self.raw_img_size[1]/self.img_size[1])
        self.strvar.set("%s loaded. (H: %d, W: %d)"%(imgfile, self.raw_img_size[0], self.raw_img_size[1]))
        self.clear_model_cache()
        self.clear_gui_cache()
        self.delete_all_tags()
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
                                    fill="",tags=tag,width=self.box_width,outline=self.palette["Box"])
            self.canvas_tags.append(tag)
            self.taglbox.insert(tk.END, tag)
        else:
            self.strvar.set("Invalid Marker.")
    
    
    def delete_select_tag(self):
        index = self.taglbox.curselection()
        if len(index) == 0:
            if len(self.markers) == 0:
                self.strvar.set("You have no Markers to delete.")
            else:
                self.strvar.set("No item selected. (Select one in the Marker List)")
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
        self.clear_tmp_markers()
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
            if self.img_size[1] <= self.canvas_size[1]:
                see_x = 0
            else:
                see_x = (marker['coord'][0]-self.canvas_size[1])/(self.img_size[1] - self.canvas_size[1])
            if self.img_size[0] <= self.canvas_size[0]:
                see_y = 0
            else:
                see_y = (marker['coord'][1]-self.canvas_size[0])/(self.img_size[0] - self.canvas_size[0])
            see_x = max(see_x,0)
            see_y = max(see_y,0)
            self.canvas.xview_moveto(see_x)
            self.canvas.yview_moveto(see_y)
        elif marker['type'] == PromptType.Box.value:
            self.canvas.create_rectangle(marker['coord'][0],marker['coord'][1],
                                        marker['coord'][2],marker['coord'][3],
                                    fill="",outline=self.palette["Highlight"],tags=tag,width=self.box_width)
            self.canvas_tmp_tags.append(tag)
            see_x = (marker['coord'][0]-self.canvas_size[1])/(self.img_size[1] - self.canvas_size[1])
            see_y = (marker['coord'][1]-self.canvas_size[0])/(self.img_size[0] - self.canvas_size[0])
            see_x = max(see_x,0)
            see_y = max(see_y,0)
            self.canvas.xview_moveto(see_x)
            self.canvas.yview_moveto(see_y)
        self.strvar.set("Choose tag {} at {}".format(self.canvas_tags[idx],marker['coord']))
        
    def highlight_mask(self, mask:np.ndarray):
        """Highlight selected mask

        Args:
            mask (np.ndarray): H x W
        """
        color = np.array(self.palette["mask"])  # (4,)
        mask_arr = np.array(mask[...,None] * color[None,None,:],dtype=np.uint8)
        mask_image = Image.fromarray(mask_arr)  # (H,W,1) * (1,1,4) -> (H,W,4)
        mask_image = mask_image.resize(sam_tools.reverse_size(self.img_size), Image.Resampling.NEAREST)
        self.masktk = ImageTk.PhotoImage(mask_image)
        self.canvas.delete("mask")
        self.canvas.create_image(0,0,anchor="nw", image=self.masktk,tag="mask")
    
    def clear_model_cache(self):
        self.masks = None
        self.scores = np.zeros(3)
        self.logits = None
        self.strvar.set("Mask Input Cache Cleared.")
    
    def clear_gui_cache(self):
        self.markers.clear()
        self.canvas_tags.clear()
        self.canvas_tmp_tags.clear()
        self.PPcnt = 0
        self.NPcnt = 0
        self.Boxcnt = 0
    
    def choose_mask(self, scale_val:str):
        val = int(scale_val)
        if(self.prev_select == val-1):
            return
        self.prev_select = val-1
        if self.masks is not None:
            self.strvar.set("Choose Mask %d (Score: %f)"%(val, self.scores[self.prev_select]))
            self.highlight_mask(self.masks[self.prev_select])
    
    def predict_mask(self, event:tk.Event=None):
        if self.sam_model is None:
            messagebox.showwarning(title="Empty Model",message="Click 'Load Model' Button First to load SAM.")
            return
        if self.input_img_arr is None:
            messagebox.showwarning(title="Empty Image",message="Click an Image file in the img listbox.")
            return
        if self.input_img_file != self.curr_img_file:
            self.clear_model_cache()
            self.input_img_file = self.curr_img_file
            self.strvar.set("Embedding New Image using %s on %s"%(self.sam_info["model"], self.sam_info["device"]))
            self.root.update()
            self.sam_model.set_image(self.input_img_arr, "RGB")  # retrive first 3 channels of image (ignore alpha channel if has)
        else:
            if not self.sam_model.is_image_set:
                self.strvar.set("Loading Cached Input Image to SAM")
                self.root.update()
                self.sam_model.set_image(self.input_img_arr, "RGB")
            self.strvar.set("Predicting by SAM %s on %s (Use Cached Image)"%(self.sam_info["model"], self.sam_info["device"]))
            self.root.update()
        input_points, point_labels, input_boxes = self.marker_to_prompts(self.markers)
        if self.logits is None:
            mask_input = None
        else:
            mask_input = self.logits[[self.prev_select],...]
        self.masks, self.scores, self.logits = sam_tools.mask_predict(self.sam_model,input_points, point_labels, input_boxes, mask_input, True)
        max_score_id = np.argmax(self.scores)
        self.scale.set(max_score_id+1)
        self.prev_select = max_score_id
        self.strvar.set("Predict Completed. (Best Mask: %d, Score: %f)"%(self.prev_select+1, self.scores[self.prev_select]))
        self.highlight_mask(self.masks[max_score_id,...])
        
    def add_marker(self, event:tk.Event):
        mx = self.canvas.canvasx(event.x)
        my = self.canvas.canvasy(event.y)
        real_coord = self.coord_tran([mx,my])
        if self.prompt_type is None:
            self.strvar.set("Choose a Prompt Type First")
        elif self.prompt_type == PromptType.PointP:
            self.add_PP(mx, my)
            self.strvar.set("Positive Point Added (%d, %d)"%(real_coord[0], real_coord[1]))
        elif self.prompt_type == PromptType.PointN:
            self.add_NP(mx, my)
            self.strvar.set("Negative Point Added (%d, %d)"%(real_coord[0], real_coord[1]))
        elif self.prompt_type == PromptType.Box:
            if self.box_state == BoxState.Release:
                self.box_state = BoxState.Hanging
                self.add_box_first_vertex(mx, my)
            elif self.box_state == BoxState.Hanging:
                self.box_state = BoxState.Release
                self.add_box_second_vertex(mx, my)
                
        if(self.curr_img_file is None):
            self.strvar.set("You Prompt before image loading! (load an image and press 'Flush' button to fix it)")

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
    
    def add_box_first_vertex(self, mx:int, my:int):
        self.canvas_tmp_tags.append("Box-Vertex1")
        self.canvas.create_oval(mx-self.point_size,my-self.point_size,
                                mx+self.point_size,my+self.point_size,
                                tag="Box-Vertex1",fill=self.palette["Box"])
        self.box_buff = [mx,my]
        self.strvar.set("First Vertex of the Box Added (%d,%d)"%(mx,my))
        
    def add_box_second_vertex(self, mx:int, my:int):
        self.clear_tmp_markers()
        marker = {"type":PromptType.Box.value, "coord":[self.box_buff[0], self.box_buff[1], mx, my]}
        self.markers.append(marker)
        self.add_marker_to_canvas(marker)
        
    def adding_rectangle(self, event:tk.Event=None):
        if self.box_buff is None or self.box_state != BoxState.Hanging:
            return
        self.canvas.delete("rectangle")
        self.canvas_tmp_tags.append("rectangle")
        mx = self.canvas.canvasx(event.x)
        my = self.canvas.canvasy(event.y)
        self.canvas.create_rectangle(self.box_buff[0],self.box_buff[1],
                                     mx,my,tags="rectangle",fill="",width=self.box_width,outline=self.palette["BoxHaning"])
        self.strvar.set("Rectangle: %d, %d, %d, %d"%(self.box_buff[0],self.box_buff[1],mx,my))
        self.root.update()
    
    def cancel_adding_rectangle(self, event:tk.Event=None):
        self.box_buff = None
        self.box_state = BoxState.Release
        self.clear_tmp_markers()
    
    def clear_tmp_markers(self):
        for tag in self.canvas_tmp_tags:
            self.canvas.delete(tag)
        self.canvas_tmp_tags.clear()

if __name__ == "__main__":
    config = yaml.load(open("config.yml",'r'),Loader=yaml.SafeLoader)
    gui = GUI(config["gui"]["icon"], config["io"]["logit_size"], config["gui"]["canvas_size"],
              config["gui"]["file_list_size"], config["gui"]["marker_list_size"], config["io"]["image_size"], config["gui"]["marker_size"],
              config["gui"]["label_font"], config["gui"]["button_font"], config["gui"]["item_font"], config["gui"]["palette"],
              config["io"]["image_dir"], config["io"]["mask_dir"], config["sam"])
    gui.run()  # Program Block
    config["sam"]["model"] = gui.sam_info["model"]
    config["sam"]["checkpoint"] = gui.sam_info["checkpoint"]
    config["io"]["image_dir"] = gui.load_dir
    config["io"]["mask_dir"] = gui.save_dir
    if gui.overwrite_config_var.get() > 0:
        yaml.dump(config, open("config.yml",'w'))
