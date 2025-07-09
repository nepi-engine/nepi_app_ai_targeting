#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus <https://www.numurus.com>.
#
# This file is part of nepi applications (nepi_apps) repo
# (see https://https://github.com/nepi-engine/nepi_apps)
#
# License: nepi applications are licensed under the "Numurus Software License", 
# which can be found at: <https://numurus.com/wp-content/uploads/Numurus-Software-License-Terms.pdf>
#
# Redistributions in source code must retain this top-level comment block.
# Plagiarizing this software to sidestep the license obligations is illegal.
#
# Contact Information:
# ====================
# - mailto:nepi@numurus.com
#

import os
# ROS namespace setup
NEPI_BASE_NAMESPACE = '/nepi/s2x/'
os.environ["ROS_NAMESPACE"] = NEPI_BASE_NAMESPACE[0:-1] # remove to run as automation script

import time
import sys
import numpy as np
import cv2
import copy
import threading
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nepi_sdk import nepi_sdk
from nepi_sdk import nepi_utils
from nepi_sdk import nepi_pc 
from nepi_sdk import nepi_img 

from std_msgs.msg import UInt8, Int32, Float32, Empty, String, Bool, Header
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge
from nepi_interfaces.msg import BoundingBox, BoundingBoxes, BoundingBox3D, BoundingBoxes3D, \
                                    ObjectCount, ClassifierSelection, \
                                    StringArray, TargetLocalization, TargetLocalizations
from nepi_interfaces.srv import ImageClassifierStatusQuery, ImageClassifierStatusQueryRequest
from nepi_interfaces.msg import Frame3DTransform
from nepi_app_ai_targeting.msg import AiTargetingStatus, AiTargetingTargets


from nepi_api.node_if import NodeClassIF
from nepi_api.messages_if import MsgIF
from nepi_api.connect_node_if import ConnectNodeClassIF
from nepi_api.system_if import SaveDataIF
from nepi_api.system_if import SaveCfgIF
from nepi_api.data_if import ImageIF

# Do this at the end
#from scipy.signal import find_peaks

#########################################
# Node Class
#########################################

class NepiAiTargetingApp(object):
  AI_MANAGER_NODE_NAME = "ai_detector_mgr"

  UDATE_PROCESS_DELAY = 1
  IMG_PUB_PROCESS_DELAY = 0.2

  #Set Initial Values
  FACTORY_FOV_VERT_DEG=70 # Camera Vertical Field of View (FOV)
  FACTORY_FOV_HORZ_DEG=110 # Camera Horizontal Field of View (FOV)
  FACTORY_TARGET_BOX_SIZE_PERCENT=80 # percent ajustment on detection box around center to use for range calc
  FACTORY_TARGET_DEPTH_METERS=0.5 # Sets the depth filter around mean depth to use for range calc
  FACTORY_TARGET_MIN_POINTS=10 # Sets the minimum number of valid points to consider for a valid range
  FACTORY_TARGET_MIN_PX_RATIO=0.1 # Sets the minimum px range between two detections, largest box between two is selected
  FACTORY_TARGET_MIN_DIST_METERS=0.0 # Sets the minimum distance between to detections, closer target is selected
  FACTORY_TARGET_MAX_AGE_SEC=10 # Remove lost targets from dictionary if older than this age
  FACTORY_USE_LAST_IMAGE = False
  FACTORY_OUTPUT_IMAGE = "Targeting_Image"

  ZERO_TRANSFORM = [0,0,0,0,0,0,0]

  NONE_CLASSES_DICT = dict()
  NONE_CLASSES_DICT["None"] = {'depth': FACTORY_TARGET_DEPTH_METERS}

  EMPTY_TARGET_DICT = dict()
  EMPTY_TARGET_DICT["None"] = {     
    'class_name': 'None', 
    'target_uid': 'None',
    'bounding_box': [0,0,0,0],
    'range_bearings': [0,0,0],
    'center_px': [0,0],
    'velocity_pxps': [0,0],
    'enter_m': [0,0,0],
    'velocity_mps': [0,0,0],
    'last_detection_timestamp': nepi_sdk.ros_duration(0)                              
    }


  node_if = None

  targeting_running = False
  data_products = ["targeting_image","target_boxes_2d","target_boxes_3d","target_localizations"]
  
  current_classifier = "None"
  current_classifier_state = "None"
  classes_list = []
  current_classifier_classes = "[]"


  cv2_bridge = CvBridge()
  current_image_topic = "None"
  current_image_header = Header()
  image_source_topic = ""
  img_width = 600
  img_height = 400
  image_sub = None
  last_img_msg = None

  depth_map_topic = "None"
  depth_map_header = Header()
  depth_map_sub = None
  np_depth_array_m = None
  has_depth_map = False
  pointcloud_topic = "None"
  has_pointcloud = False

  bbs_msg = None
  bb3s_msg = None
  target_box_3d_list = None
  class_color_list = []

  last_targeting_enable = False
  last_image_topic = "None"
  
  selected_classes_dict = dict()
  current_targets_dict = dict()
  active_targets_dict = dict()
  lost_targets_dict = dict()
  selected_target = "None"
  classes_selected = False

  targeting_class_list = []
  targeting_target_list = []
  detection_image_pub = None


  classifier_running = False
  classifier_loading_progress = 0.0
  classifier_threshold = 0.3

  last_snapshot = time.time()
  last_cv2_img = None

  no_object_count = 0

  reset_image_topic = False
  app_enabled = False
  app_msg = "App not enabled"
  
  img_acquire = False
  img_msg = None
  last_img_msg = None
  img_lock = threading.Lock()


  target_locs_acquire = False
  target_locs = []
  target_locs_lock = threading.Lock()

  img_has_subs = False

  last_app_enabled = False
  app_enabled = False
  image_fov_vert = self.FACTORY_FOV_VERT_DEG
  image_fov_horz = self.self.FACTORY_FOV_HORZ_DEG
  last_classifier = ""
  selected_classes_dict = []
  target_box_percent = self.FACTORY_TARGET_BOX_SIZE_PERCENT
  default_target_depth = self.FACTORY_TARGET_DEPTH_METERS
  target_min_points = self.FACTORY_TARGET_MIN_POINTS
  target_min_px_ratio = self.FACTORY_TARGET_MIN_PX_RATIO
  target_min_dist_m = self.FACTORY_TARGET_MIN_DIST_METERS
  target_age_filter = self.FACTORY_TARGET_MAX_AGE_SEC
  frame_3d_transform = self.ZERO_TRANSFORM
  
  #######################
  ### Node Initialization
  DEFAULT_NODE_NAME = "app_ai_targeting" # Can be overwitten by luanch command
  def __init__(self):
    #### APP NODE INIT SETUP ####
    nepi_sdk.init_node(name= self.DEFAULT_NODE_NAME)
    self.class_name = type(self).__name__
    self.base_namespace = nepi_sdk.get_base_namespace()
    self.node_name = nepi_sdk.get_node_name()
    self.node_namespace = nepi_sdk.get_node_namespace()

    ##############################  
    # Create Msg Class
    self.msg_if = MsgIF(log_name = self.class_name)
    self.msg_if.pub_info("Starting IF Initialization Processes")

    ##############################     
    # Initialize Class Variables


    ##############################
    ### Setup Node


    # Configs Config Dict ####################
    self.CFGS_DICT = {
            'init_callback': self.initCb,
            'reset_callback': self.resetCb,
            'factory_reset_callback': self.factoryResetCb,
            'init_configs': True,
            'namespace': self.node_namespace
    }

    # Params Config Dict ####################
    self.PARAMS_DICT = {
        'app_enabled': {
            'namespace': self.node_namespace,
            'factory_val': False
        },
        'image_fov_vert': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_FOV_VERT_DEG
        },
        'image_fov_horz': {
            'namespace': self.node_namespace,
            'factory_val': self.self.FACTORY_FOV_HORZ_DEG
        },
        'last_classifier': {
            'namespace': self.node_namespace,
            'factory_val': ""
        },
        'selected_classes_dict': {
            'namespace': self.node_namespace,
            'factory_val': []
        },
        'target_box_percent': {
            'namespace': self.target_box_percent,
            'factory_val': self.FACTORY_TARGET_BOX_SIZE_PERCENT
        },
        'default_target_depth': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_TARGET_DEPTH_METERS
        },
        'target_min_points': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_TARGET_MIN_POINTS
        },
        'target_min_px_ratio': {
            'namespace': self.target_box_percent,
            'factory_val': self.FACTORY_TARGET_MIN_PX_RATIO
        },
        'target_min_dist_m': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_TARGET_MIN_DIST_METERS
        },
        'target_age_filter': {
            'namespace': self.node_namespace,
            'factory_val': self.FACTORY_TARGET_MAX_AGE_SEC
        },
        'frame_3d_transform': {
            'namespace': self.node_namespace,
            'factory_val': self.ZERO_TRANSFORM
        }

    }

    # Publishers Config Dict ####################
    self.PUBS_DICT = {
        'status': {
            'namespace': self.node_namespace,
            'topic': 'status',
            'msg': AiTargetingStatus,
            'qsize': 1,
            'latch': True
        },
        'targets': {
            'namespace': self.node_namespace,
            'topic': 'targets',
            'msg': AiTargetingTargets,
            'qsize': 1,
            'latch': True
        },
        'boxes_count': {
            'namespace': self.node_namespace,
            'topic': 'boxes_count',
            'msg': ObjectCount,
            'qsize': 1,
            'latch': True
        },
        'boxes3d_count': {
            'namespace': self.node_namespace,
            'topic': 'boxes3d_count',
            'msg': ObjectCount,
            'qsize': 1,
            'latch': True
        },
        'target_count': {
            'namespace': self.node_namespace,
            'topic': 'target_count',
            'msg': ObjectCount,
            'qsize': 1,
            'latch': True
        },
        'target_boxes_2d': {
            'namespace': self.node_namespace,
            'topic': 'target_boxes_2d',
            'msg': BoundingBoxes,
            'qsize': 1,
            'latch': True
        },
        'target_boxes_3d': {
            'namespace': self.node_namespace,
            'topic': 'target_boxes_3d',
            'msg': BoundingBoxes3D,
            'qsize': 1,
            'latch': True
        },
        'target_localizations': {
            'namespace': self.node_namespace,
            'topic': 'target_localizations',
            'msg': TargetLocalizations,
            'qsize': 1,
            'latch': True
        }

    }

    # Subscribers Config Dict ####################
    self.SUBS_DICT = {
        'publish_status': {
            'namespace': self.node_namespace,
            'topic': 'publish_status',
            'msg': Empty,
            'qsize': 10,
            'callback': self.pubStatusCb, 
            'callback_args': ()
        },
        'enable_app': {
            'namespace': self.node_namespace,
            'topic': 'enable_app',
            'msg': Bool,
            'qsize': 10,
            'callback': self.appEnableCb, 
            'callback_args': ()
        },
        'set_image_fov_vert': {
            'namespace': self.node_namespace,
            'topic': 'set_image_fov_vert',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setVertFovCb, 
            'callback_args': ()
        },
        'set_image_fov_horz': {
            'namespace': self.node_namespace,
            'topic': 'set_image_fov_horz',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setHorzFovCb, 
            'callback_args': ()
        },
        'add_all_target_classes': {
            'namespace': self.node_namespace,
            'topic': 'add_all_target_classes',
            'msg': Empty,
            'qsize': 10,
            'callback': self.addAllClassesCb, 
            'callback_args': ()
        },
        'add_target_class': {
            'namespace': self.node_namespace,
            'topic': 'add_target_class',
            'msg': String,
            'qsize': 10,
            'callback': self.addClassCb, 
            'callback_args': ()
        },
        'remove_target_class': {
            'namespace': self.node_namespace,
            'topic': 'remove_target_class',
            'msg': String,
            'qsize': 10,
            'callback': self.removeClassCb, 
            'callback_args': ()
        },
        'select_target': {
            'namespace': self.node_namespace,
            'topic': 'select_target',
            'msg': String,
            'qsize': 10,
            'callback': self.selectTargetCb, 
            'callback_args': ()
        },
        'set_target_box_size_percent': {
            'namespace': self.node_namespace,
            'topic': 'set_target_box_size_percent',
            'msg': Int32,
            'qsize': 10,
            'callback': self.setTargetBoxPercentCb, 
            'callback_args': ()
        },
        'set_default_target_detpth': {
            'namespace': self.node_namespace,
            'topic': 'set_default_target_detpth',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setDefaultTargetDepthCb, 
            'callback_args': ()
        },
        'set_target_min_points': {
            'namespace': self.node_namespace,
            'topic': 'set_target_min_points',
            'msg': Int32,
            'qsize': 10,
            'callback': self.setTargetMinPointsCb, 
            'callback_args': ()
        },
        'set_target_min_px_ratio': {
            'namespace': self.node_namespace,
            'topic': 'set_target_min_px_ratio',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setTargetMinPxRatioCb, 
            'callback_args': ()
        },
        'set_age_filter': {
            'namespace': self.node_namespace,
            'topic': 'set_age_filter',
            'msg': Float32,
            'qsize': 10,
            'callback': self.setAgeFilterCb, 
            'callback_args': ()
        },
        'set_frame_3d_transform': {
            'namespace': self.node_namespace,
            'topic': 'set_frame_3d_transform',
            'msg': Frame3DTransform,
            'qsize': 10,
            'callback': self.setFrame3dTransformCb, 
            'callback_args': ()
        },
        'clear_frame_3d_transform': {
            'namespace': self.node_namespace,
            'topic': 'clear_frame_3d_transform',
            'msg': Empty,
            'qsize': 10,
            'callback': self.clearFrame3dTransformCb, 
            'callback_args': ()
        }
        'found_object': {
            'namespace': self.node_namespace,
            'topic': '/found_object' #self.ai_mgr_namespace  + "/found_object"
            'msg': ObjectCount,
            'qsize': 1,
            'callback': self.foundObjectCb, 
            'callback_args': ()
        },
        'bounding_boxes': {
            'namespace': self.node_namespace,
            'topic': '/bounding_boxes' self.ai_mgr_namespace  + "/bounding_boxes"
            'msg': BoundingBoxes,
            'qsize': 1,
            'callback': self.objectDetectedCb, 
            'callback_args': ()
        },
        'image_topic': {
            'namespace': self.node_namespace,
            'topic': '/image_topic'
            'msg': Image,
            'qsize': 1,
            'callback': self.imageCb, 
            'callback_args': ()
        },
        'depth_map_topic': {
            'namespace': self.node_namespace,
            'topic': '/image_topic'
            'msg': Image,
            'qsize': 10,
            'callback': self.depthMapCb, 
            'callback_args': ()
        }
    }

    # Create Node Class ####################
    self.node_if = NodeClassIF(
                    configs_dict = self.CFGS_DICT,
                    params_dict = self.PARAMS_DICT,
                    pubs_dict = self.PUBS_DICT,
                    subs_dict = self.SUBS_DICT
    )

    ready = self.node_if.wait_for_ready()


    # Setup Image IF
    self.image_if = ImageIF(namespace = self.node_namespace, topic = 'targeting_image')




    self.ai_mgr_namespace = os.path.join(self.base_namespace, self.AI_MANAGER_NODE_NAME)
   


    ##############################
    self.initCb(do_updates = True)

    # Set up save data and save config services 
    factory_data_rates= {}
    for d in self.data_products:
        factory_data_rates[d] = [0.0, 0.0, 100.0] # Default to 0Hz save rate, set last save = 0.0, max rate = 100.0Hz
    if 'targeting_image' in self.data_products:
        factory_data_rates['targeting_image'] = [1.0, 0.0, 100.0] 
    self.save_data_if = SaveDataIF(data_products = self.data_products_list, factory_rate_dict = factory_data_rates)


    ##############################
    # Message Image to publish when detector not running
    message = "APP NOT ENABLED"
    cv2_img = nepi_img.create_message_image(message)
    self.app_ne_img = nepi_img.cv2img_to_rosimg(cv2_img)
    self.app_ne_img.header.stamp = nepi_sdk.get_msg_time()
    if self.node_if is not None:
      self.node_if.publish_pub('image_pub',self.app_ne_img)

    message = "WAITING FOR AI DETECTOR TO START"
    cv2_img = nepi_img.create_message_image(message)
    self.classifier_nr_img = nepi_img.cv2img_to_rosimg(cv2_img)

    message = "WAITING FOR TARGET CLASSES SELECTION"
    cv2_img = nepi_img.create_message_image(message)
    self.no_class_img = nepi_img.cv2img_to_rosimg(cv2_img)

    ##############################
    # Get AI Manager Service Call
    ##AI_MGR_STATUS_SERVICE_NAME = self.ai_mgr_namespace  + "/img_classifier_status_query"
    #self.AI_MGR_STATUS_SERVICE_NAME = nepi_sdk.connect_service(AI_MGR_STATUS_SERVICE_NAME, ImageClassifierStatusQuery)
    # Start AI Manager Subscribers
    FOUND_OBJECT_TOPIC = self.ai_mgr_namespace  + "/found_object"
    self.nepi_sdk.create_subscriber(FOUND_OBJECT_TOPIC, ObjectCount, self.foundObjectCb, queue_size = 1)
    BOUNDING_BOXES_TOPIC = self.ai_mgr_namespace  + "/bounding_boxes"
    self.nepi_sdk.create_subscriber(BOUNDING_BOXES_TOPIC, BoundingBoxes, self.objectDetectedCb, queue_size = 1)
    time.sleep(1)

    # Set up timer callbacks
    nepi_sdk.timer(nepi_sdk.ros_duration(self.UDATE_PROCESS_DELAY), self.updaterCb)
    nepi_sdk.timer(nepi_sdk.ros_duration(self.IMG_PUB_PROCESS_DELAY), self.imagePubCb)


    ## Initiation Complete
    self.msg_if.pub_info(" Initialization Complete")
    self.publish_status()
    self.publish_targets()

    # Spin forever (until object is detected)
    nepi_sdk.spin()




  #######################
  ### App Config Functions

  def initCb(self,do_updates = False):
      if self.node_if is not None:
         
        self.app_enabled = self.node_if.get_param('app_enabled')
        self.image_fov_vert = self.node_if.get_param('image_fov_vert')
        self.image_fov_horz = self.node_if.get_param('image_fov_horz')
        self.selected_classes_dict = self.node_if.get_param('selected_classes_dict')
        self.target_box_percent = self.node_if.get_param('target_box_percent')
        self.default_target_depth = self.node_if.get_param('default_target_depth')
        self.target_min_points = self.node_if.get_param('target_min_points')
        self.target_min_px_ratio = self.node_if.get_param('target_min_px_ratio')
        self.target_min_dist_m = self.node_if.get_param('target_min_dist_m')
        self.target_age_filter = self.node_if.get_param('target_age_filter')
        self.frame_3d_transform = self.node_if.get_param('frame_3d_transform')

      if do_updates == True:
        pass
      self.publish_status()

  def resetCb(self,do_updates = True):
      self.msg_if.pub_warn("Reseting")
      if self.node_if is not None:
        pass
      if do_updates == True:
        pass
      self.initCb(do_updates = do_updates)


  def factoryResetCb(self,do_updates = True):
      self.msg_if.pub_warn("Factory Reseting")
      if self.node_if is not None:
        pass
      if do_updates == True:
        pass
      self.initCb(do_updates = do_updates)


  ###################
  ## Status Publisher
  def publish_status(self):
    status_msg = AiTargetingStatus()

    status_msg.app_enabled = self.app_enabled
    status_msg.app_msg = self.app_msg

    status_msg.image_topic = self.current_image_topic
    status_msg.image_fov_vert_degs = self.image_fov_vert
    status_msg.image_fov_horz_degs = self.image_fov_horz

    status_msg.has_depth_map = self.has_depth_map
    status_msg.depth_map_topic = self.depth_map_topic
    status_msg.has_pointcloud = self.has_pointcloud
    status_msg.pointcloud_topic = self.pointcloud_topic

    status_msg.classifier_running = self.classifier_running

    avail_classes = self.classes_list
    #self.msg_if.pub_warn(" available classes: " + str(avail_classes))
    if len(avail_classes) == 0:
      avail_classes = ["None"]
    avail_classes = sorted(avail_classes)
    status_msg.available_classes_list = avail_classes
    selected_classes_dict = self.selected_classes_dict
    purge_class_list = []
    for key in selected_classes_dict.keys():
      if key not in avail_classes:
        purge_class_list.append(key)
    for key in purge_class_list:
      del selected_classes_dict[key]
    sel_classes_list = []
    depth_list = []
    for key in selected_classes_dict.keys():
      sel_classes_list.append(key)
      depth_list.append((selected_classes_dict[key]['depth']))
    if len(sel_classes_list) == 0:
      sel_classes_list = ['None']
      depth_list = [0]
    status_msg.selected_classes_list = (sel_classes_list)
    status_msg.selected_classes_depth_list = (depth_list)


    status_msg.target_box_size_percent = self.target_box_percent
    status_msg.default_target_depth_m = self.default_target_depth
    status_msg.target_min_points = self.target_min_points
    status_msg.target_min_px_ratio = self.target_min_px_ratio
    status_msg.target_min_dist_m = self.target_min_dist_m
    status_msg.target_age_filter = self.target_age_filter
    # The transfer frame for target data adjustments from image's native frame to the nepi center frame
    transform = self.frame_3d_transform
    transform_msg = Frame3DTransform()
    transform_msg.translate_vector.x = transform[0]
    transform_msg.translate_vector.y = transform[1]
    transform_msg.translate_vector.z = transform[2]
    transform_msg.rotate_vector.x = transform[3]
    transform_msg.rotate_vector.y = transform[4]
    transform_msg.rotate_vector.z = transform[5]
    transform_msg.heading_offset = transform[6]
    status_msg.frame_3d_transform = transform_msg

    if self.node_if is not None:
      self.node_if.publish_pub('status_pub', status_msg)

 
  ## Status Publisher
  def publish_targets(self):
    targets_ms = AiTargetingTargets()

    targets_list = self.active_targets_dict.keys()
    avail_targets_list = []
    if self.selected_target != "None" and self.selected_target not in targets_list:
      avail_targets_list.append(self.selected_target)
    for target in targets_list:
      avail_targets_list.append(target) 
    avail_targets_list = sorted(avail_targets_list)
    avail_targets_list.insert(0,"None")
    targets_ms.available_targets_list = (avail_targets_list)
    targets_ms.selected_target = self.selected_target
    #self.msg_if.pub_warn(" Targets Msg: " + str(targets_ms))
    if self.node_if is not None:
      self.node_if.publish_pub('targets_pub', targets_ms)     
    
 

  def updaterCb(self,timer):
    self.last_image_topic = self.current_image_topic
    update_status = False
    app_enabled = self.app_enabled
    app_msg = ""
    if app_enabled == False:
      self.target_detected = False
      app_msg += "App not enabled"
      if self.image_sub is not None:
        self.msg_if.pub_warn(" App Disabled, Unsubscribing from Image topic : " + self.last_image_topic)
        self.image_sub.unregister()
        time.sleep(1)
        self.image_sub = None
    elif self.last_app_enabled != app_enabled:
      update_status = True
    self.last_app_enabled = app_enabled

    # Update classifier info
    ai_mgr_status_response = None
    try:
      ai_mgr_status_response = self.get_ai_mgr_status_service()
      #self.msg_if.pub_info(" Got classifier status  " + str(ai_mgr_status_response))
    except Exception as e:
      self.msg_if.pub_warn("Failed to call AI MGR STATUS service" + str(e))
      self.classifier_running = False
      self.last_classiier = ""
      self.publish_status()
      if self.node_if is not None:
        self.node_if.set_param('last_classiier', "")
      app_msg += ", AI Detector not connected"
    if ai_mgr_status_response != None:
      app_msg += ", AI Detector connected"
      #status_str = str(ai_mgr_status_response)
      #self.msg_if.pub_warn(" got ai manager status: " + status_str)
      self.current_image_topic = ai_mgr_status_response.selected_img_topic
      self.current_classifier = ai_mgr_status_response.selected_classifier
      self.current_classifier_state = ai_mgr_status_response.classifier_state
      self.classifier_running = self.current_classifier_state == "Running"
      classes_list = ai_mgr_status_response.selected_classifier_classes
      if classes_list != self.classes_list:
        update_status = True
        self.classes_list = classes_list
        if len(self.classes_list) > 0:
          cmap = plt.get_cmap('viridis')
          color_list = cmap(np.linspace(0, 1, len(self.classes_list))).tolist()
          rgb_list = []
          for color in color_list:
            rgb = []
            for i in range(3):
              rgb.append(int(color[i]*255))
            rgb_list.append(rgb)
          self.class_color_list = rgb_list
      self.classes_list = classes_list
      self.last_classiier = self.current_classifier
      self.publish_status()
      if self.node_if is not None:
        self.node_if.set_param('last_classiier', self.current_classifier)
      #self.msg_if.pub_warn(" Got image topics last and current: " + self.last_image_topic + " " + self.current_image_topic)

      # Update Image Topic Subscriber
      if self.classifier_running == False:
        app_msg += ", Classifier not running"
        self.target_detected = False
        self.current_image_topic = "None"
      else:
        app_msg += ", Classifier running"
        if (self.last_image_topic != self.current_image_topic) or (self.image_sub == None and self.current_image_topic != "None") or self.reset_image_topic == True:
          update_status = True
          self.reset_image_topic = False
          image_topic = nepi_sdk.find_topic(self.current_image_topic)
          if image_topic == "":
            self.msg_if.pub_warn(" Could not find image update topic: " + self.current_image_topic)
          elif app_enabled == True and image_topic != "None":
            self.msg_if.pub_info(" Found detect Image update topic : " + image_topic)
            if self.image_sub != None:
              self.msg_if.pub_warn(" Unsubscribing to Image topic : " + self.last_image_topic)
              self.image_sub.unregister()
              time.sleep(1)
              self.image_sub = None

            self.msg_if.pub_info(" Subscribing to Image topic : " + image_topic)
            self.image_sub = self.nepi_sdk.create_subscriber(image_topic, Image, self.imageCb, queue_size = 1)

            # Look for Depth Map
            depth_map_topic = self.current_image_topic.rsplit('/',1)[0] + "/depth_map"
            depth_map_topic = nepi_sdk.find_topic(depth_map_topic)
            if depth_map_topic == "":
              depth_map_topic = "None"
              self.has_depth_map = False
            else:
              self.has_depth_map = True
            self.depth_map_topic = depth_map_topic
            #self.msg_if.pub_warn(self.depth_map_topic)
            if depth_map_topic != "None":
              if self.depth_map_sub != None:
                self.depth_map_sub.unregister()
                self.depth_map_sub = None
                time.sleep(1)
              self.msg_if.pub_info(" Subscribing to Depth Map topic : " + depth_map_topic)
              self.depth_map_sub = self.nepi_sdk.create_subscriber(depth_map_topic, Image, self.depthMapCb, queue_size = 10)
              update_status = True
              
              # If there is a depth_map, check for pointdcloud
              pointcloud_topic = self.current_image_topic.rsplit('/',1)[0] + "/pointcloud"
              pointcloud_topic = nepi_sdk.find_topic(pointcloud_topic)
              if pointcloud_topic == "":
                pointcloud_topic = "None"
                self.has_pointcloud = False
              else:
                self.has_pointcloud = True
              self.pointcloud_topic = pointcloud_topic
              update_status = True
          else:
            self.last_image_topic = ""

          if self.current_image_topic == "None" or self.current_image_topic == "":  # Reset last image topic
            if self.image_sub != None:
              self.msg_if.pub_warn(" Unsubscribing to Image topic : " + self.current_image_topic)
              self.image_sub.unregister()
              time.sleep(1)
              self.image_sub = None
              update_status = True




      # Print a message image if needed

      selected_classes_dict = self.selected_classes_dict
      classes_sel = False
      for key in selected_classes_dict.keys():
        if key in self.classes_list:
          classes_sel = True
      self.classes_selected = classes_sel

      if app_enabled == False:
        #self.msg_if.pub_warn("Publishing Not Enabled image")
        if not nepi_sdk.is_shutdown():
          self.app_ne_img.header.stamp = nepi_sdk.get_msg_time()
          self.image_if.publish_cv2_image(self.app_ne_img)
      elif self.classifier_running == False:
        if not nepi_sdk.is_shutdown():
          self.classifier_nr_img.header.stamp = nepi_sdk.get_msg_time()
          self.image_if.publish_cv2_image(self.classifier_nr_img)
      elif self.classes_selected == False:
        if not nepi_sdk.is_shutdown():
          self.no_class_img.header.stamp = nepi_sdk.get_msg_time()
          self.image_if.publish_cv2_image(self.no_class_img)


      # Update Current Targets List based on Age and Publish
      current_timestamp = nepi_sdk.get_msg_time()
      active_targets_dict = copy.deepcopy(self.active_targets_dict)
      lost_targets_dict = copy.deepcopy(self.lost_targets_dict)
      purge_list = []
      age_filter_sec = self.target_age_filter
      #self.msg_if.pub_warn(active_targets_dict)
      for target in active_targets_dict.keys():
        last_timestamp = active_targets_dict[target]['last_detection_timestamp']
        #self.msg_if.pub_warn(target)
        #self.msg_if.pub_warn(get_msg_timestamp.to_sec())
        #self.msg_if.pub_warn(last_timestamp.to_sec())
        age =(current_timestamp.to_sec() - last_timestamp.to_sec())
        #self.msg_if.pub_warn("Target " + target + " age: " + str(age))
        if age > age_filter_sec:
          purge_list.append(target)
      #self.msg_if.pub_warn("Target Purge List: " + str(purge_list))
      for target in purge_list: 
          lost_targets_dict[target] = active_targets_dict[target]
          self.msg_if.pub_info(" Purging target: " + target + " from active target list")
          del active_targets_dict[target]
      self.active_targets_dict = active_targets_dict
      self.lost_targets_dict = lost_targets_dict
      self.publish_targets()
    
    
    # Additional Cleanup steps
    if self.classifier_running == False or self.current_image_topic == "None" or self.current_image_topic == "":  # Turn off targeting subscribers and reset last image topic
      self.targeting_running = False
      self.current_targets_dict = dict()
      if self.image_sub != None:
        self.msg_if.pub_warn(" Unsubscribing to Image topic : " + self.current_image_topic)
        self.image_sub.unregister()
        self.image_sub = None
        update_status = True
      if self.depth_map_sub != None:
        self.depth_map_sub.unregister()
        self.has_depth_map = False
        self.depth_map_header = Header()
      self.depth_map_topic = "None"
      self.has_pointcloud = False
      self.pointcloud_topic = "None"
      time.sleep(1)


    self.app_msg = app_msg
    # Publish status if needed
    if update_status == True:
      self.publish_status()



  ###################
  ## AI App Callbacks
  def pubStatusCb(self,msg):
    self.publish_status()



  def appEnableCb(self,msg):
    #self.msg_if.pub_info(msg)
    val = msg.data
    self.app_enabled = val
    self.publish_status()
    if self.node_if is not None:
      self.node_if.set_param('app_enabled',val)

  def addAllClassesCb(self,msg):
    ##self.msg_if.pub_info(msg)
    classes = self.classes_list
    depth = self.default_target_depth
    selected_dict = dict()
    for Class in classes:
      selected_dict[Class] = {'depth': depth }
    self.selected_classes_dict = selected_dict
    self.publish_status()
    if self.node_if is not None:
      self.node_if.set_param('selected_classes_dict', selected_dict)

  def removeAllClassesCb(self,msg):
    ##self.msg_if.pub_info(msg)
    self.selected_classes_dict = dict()
    self.publish_status()
    if self.node_if is not None:
      self.node_if.set_param('selected_classes_dict', dict())

  def addClassCb(self,msg):
    ##self.msg_if.pub_info(msg)
    class_name = msg.data
    class_depth_m = self.default_target_depth
    if class_name in self.classes_list:
      selected_classes_dict = self.selected_classes_dict
      selected_classes_dict[class_name] = {'depth': class_depth_m}
      self.selected_classes_dict = selected_classes_dict
      self.publish_status()
      if self.node_if is not None:
        self.node_if.set_param('selected_classes_dict', selected_classes_dict)
    self.publish_status()

  def removeClassCb(self,msg):
    ##self.msg_if.pub_info(msg)
    class_name = msg.data
    selected_classes_dict = self.selected_classes_dict
    if class_name in selected_classes_dict.keys():
      del selected_classes_dict[class_name]
    self.selected_classes_dict = selected_classes_dict
    self.publish_status()
    if self.node_if is not None:
      self.node_if.set_param('selected_classes_dict', selected_classes_dict)

  def selectTargetCb(self,msg):
    ##self.msg_if.pub_info(msg)
    target_name = msg.data
    if target_name == 'None' or target_name in self.active_targets_dict.keys():
      self.selected_target = target_name
    self.publish_targets()

  def setVertFovCb(self,msg):
    ##self.msg_if.pub_info(msg)
    fov = msg.data
    if fov > 0:
      self.image_fov_vert = fov
      self.publish_status()
      if self.node_if is not None:
        self.node_if.set_param('image_fov_vert',  fov)
    self.publish_status()


  def setHorzFovCb(self,msg):
    ##self.msg_if.pub_info(msg)
    fov = msg.data
    if fov > 0:
      self.image_fov_horz = fov
      self.publish_status()
      if self.node_if is not None:
        self.node_if.set_param('image_fov_horz',  fov)
    self.publish_status()
    
  def setTargetBoxPercentCb(self,msg):
    #self.msg_if.pub_info(msg)
    val = msg.data
    if val >= 10 and val <= 200:
      self.target_box_percent = val
      self.publish_status()
      if self.node_if is not None:
        self.node_if.set_param('target_box_percent',val)
    self.publish_status()   
      
  def setDefaultTargetDepthCb(self,msg):
    #self.msg_if.pub_info(msg)
    val = msg.data
    if val >= 0:
      self.default_target_depth = val
      self.publish_status()
      if self.node_if is not None:
        self.node_if.set_param('default_target_depth',val)
    self.publish_status()   

  def setTargetMinPointsCb(self,msg):
    #self.msg_if.pub_info(msg)
    val = msg.data
    if val >= 0:
      self.target_min_points = val
      self.publish_status() 
      if self.node_if is not None:
        self.node_if.set_param('target_min_points',val)
    self.publish_status() 

  def setTargetMinPxRatioCb(self,msg):
    #self.msg_if.pub_info(msg)
    val = msg.data
    if val >= 0 and val <= 1:
      self.target_min_px_ratio = val
      self.publish_status() 
      if self.node_if is not None:
        self.node_if.set_param('target_min_px_ratio',val)
    self.publish_status() 

  def setTargetMinDistMCb(self,msg):
    #self.msg_if.pub_info(msg)
    val = msg.data
    if val >= 0:
      self.target_min_dist_m = val
      self.publish_status() 
      if self.node_if is not None:
        self.node_if.set_param('target_min_dist_m',val)
    self.publish_status() 

  def setAgeFilterCb(self,msg):
    #self.msg_if.pub_info(msg)
    val = msg.data
    if val >= 0:
      self.target_age_filter = val
      self.publish_status()
      if self.node_if is not None:
        self.node_if.set_param('target_age_filter',val)
    self.publish_status()

  def setFrame3dTransformCb(self, msg):
      new_transform_msg = msg
      self.setFrame3dTransform(new_transform_msg)
      

  def (self, transform_msg):
      #self.msg_if.pub_info("AI_TARG_APP: Recieved Transform message " + str(transform_msg))
      x = transform_msg.translate_vector.x
      y = transform_msg.translate_vector.y
      z = transform_msg.translate_vector.z
      roll = transform_msg.rotate_vector.x
      pitcself.publish_status()h = transform_msg.rotate_vector.y
      yaw = transform_msg.rotate_vector.z
      heading = transform_msg.heading_offset
      transform = [x,y,z,roll,pitch,yaw,heading]
      self.frame_3d_transform = transform
      self.publish_status()
      if self.node_if is not None:
        self.node_if.set_param('frame_3d_transform',  transform)
      #self.msg_if.pub_info("AI_TARG_APP: Updated Transform: " + str(transform))

  def clearFrame3dTransformCb(self, msg):
      new_transform_msg = msg
      self.clearFrame3dTransforsetFrame3dTransformm()

  def clearFrame3dTransform(self, transform_msg):
      transform = self.ZERO_TRANSFORM
      self.idx/frame_3d_transform = transform
      self.publish_status()
      if self.node_if is not None:
        self.node_if.set_param('idx/frame_3d_transform',  transform)
      self.status_msg.frame_3d_transform = transform_msg
      self.publishStatus(do_updates=False) # Updated inline here 

  

  #######################
  ### AI Magnager Callbacks

  ### If object(s) detected, save bounding box info to global
  def objectDetectedCb(self,bounding_boxes_msg):
    app_enabled = self.app_enabled
    get_msg_timestamp = bounding_boxes_msg.header.stamp

    if app_enabled == False:
      self.target_detected = False
      self.target_locs_lock.acquire()
      self.target_locs = []     
      self.target_locs_lock.release()
    else:

      detect_header = bounding_boxes_msg.header
      get_msg_timestamp = bounding_boxes_msg.header.stamp
      image_seq_num = bounding_boxes_msg.header.seq
      bbs_msg=copy.deepcopy(bounding_boxes_msg)
      transform = self.frame_3d_transform
      selected_classes_dict = self.selected_classes_dict

      # Process targets
      active_targets_dict = copy.deepcopy(self.active_targets_dict)
      current_targets_dict = dict()
      bbs2d = []
      tls = []
      bbs3d = []

      #self.msg_if.pub_warn("Got image hxw: " + str([self.img_height,self.img_width]))
      if self.img_height != 0 and self.img_width != 0:
          # Iterate over all of the objects and calculate range and bearing data
          
          image_fov_vert = self.image_fov_vert
          image_fov_horz = self.image_fov_horz
          target_box_adjust_percent = self.target_box_percent
          target_min_points = self.target_min_points    
          target_min_px_ratio = self.target_min_px_ratio
          target_min_dist_m = self.target_min_dist_m
          target_uids = []
          if bbs_msg is not None:
              box_class_list = []
              box_area_list = []
              box_mmx_list = []
              box_mmy_list = []
              box_center_list = []
              for i, box in enumerate(bbs_msg.bounding_boxes):
                  box_class_list.append(box.Class)
                  box_area=(box.xmax-box.xmin)*(box.ymax-box.ymin)
                  box_area_list.append(box_area)
                  box_mmx_list.append([box.xmin,box.xmax])
                  box_mmy_list.append([box.ymin,box.ymax])
                  box_y = box.ymin + (box.ymax - box.ymin)
                  box_x = box.xmin + (box.xmax - box.xmin)
                  box_center = [box_y,box_x]
                  box_center_list.append(box_center)
                  #Calculate areas
                  img_area = self.img_height * self.img_width
                  area_pixels = (box.xmax - box.xmin) * (box.ymax - box.ymin)
                  if img_area > 1:
                      area_ratio = area_pixels / img_area
                  else:
                      area_ratio = -999

              for i, box in enumerate(bbs_msg.bounding_boxes):
                  if box.Class in selected_classes_dict.keys():
                      target_depth_m = selected_classes_dict[box.Class]['depth']
                      # Get target label
                      target_label=box.Class
                      # reduce target box based on user settings
                      y_len = (box.ymax - box.ymin)
                      x_len = (box.xmax - box.xmin)
                      adj_ratio = float(target_box_adjust_percent )/100.0
                      if target_box_adjust_percent == 100: 
                          delta_y = 0
                          delta_x = 0
                      else:
                          adj = float(target_box_adjust_percent )/100.0
                          delta_y = int(y_len / 2 * (adj_ratio-1))
                          delta_x = int(x_len / 2 * (adj_ratio-1))
                      ymin_adj=box.ymin - delta_y
                      ymax_adj=box.ymax + delta_y
                      xmin_adj=box.xmin - delta_x
                      xmax_adj=box.xmax + delta_x
                      #self.msg_if.pub_warn("Got adjusted box sizes yx: " + str([ymin_adj,ymax_adj,xmin_adj,xmax_adj]))
                      # Calculate target range
                      target_range_m=float(-999)  # NEPI standard unset value
                      target_depth = selected_classes_dict[box.Class]['depth']
                      np_depth_array_m = copy.deepcopy(self.np_depth_array_m)
                      if np_depth_array_m is not None and self.depth_map_topic != "None":
                          
                          depth_map_header = copy.deepcopy(self.depth_map_header)
                          # Get target range from cropped and filtered depth data
                          depth_box_adj= np_depth_array_m[ymin_adj:ymax_adj,xmin_adj:xmax_adj]
                          depth_array=depth_box_adj.flatten()
                          depth_array = depth_array[~np.isinf(depth_array)] # remove inf entries
                          depth_array = depth_array[~np.isnan(depth_array)] # remove nan entries
                          depth_array = depth_array[depth_array>0] # remove zero entries
                          depth_val=np.mean(depth_array) # Initialize fallback value.  maybe updated
                          #self.msg_if.pub_warn("got depth data")
                          #self.msg_if.pub_warn(depth_val)
                          # Try histogram calculation
                          try:
                            min_range = np.min(depth_array)
                            max_range = np.max(depth_array)
                          except:
                            min_range = 0
                            max_range = 0
                          delta_range = max_range - min_range
                          if delta_range > target_depth/2:
                              bins_per_target = 10
                              bin_step = target_depth / bins_per_target
                              num_bins = 1
                              #self.msg_if.pub_warn('delta_range: ' + str(delta_range))
                              #self.msg_if.pub_warn('bin_step: ' + str(bin_step))
                              if bin_step > 0.001 and math.isinf(delta_range) == False :
                                num_bins = int(delta_range / bin_step)
                              # Get histogram
                              hist, hbins = np.histogram(depth_array, bins = num_bins, range = (min_range,max_range))
                              bins = hbins[1:] + (hbins[1:] - hbins[:-1]) / 2
                              peak_dist = int(bins_per_target / 2)
                              #max_hist_inds,ret_dict = find_peaks(hist, distance=peak_dist)
                              #max_hist_inds = list(max_hist_inds)
                              #max_hist_ind = max_hist_inds[0]
                              max_hist_val = hist[0]
                              max_hist_ind = 0
                              for ih, val in enumerate(hist):
                                  if val > max_hist_val:
                                      max_hist_val = val
                                      max_hist_ind = ih
                                  elif val < max_hist_val:
                                      break 
                              #self.msg_if.pub_warn(max_hist_ind)
                              hist_len = len(hist)
                              bins_len = len(bins)
                              # Hanning window on targets
                              win_len = bins_per_target
                              if hist_len > win_len:
                                  win_len_half = int(bins_per_target/2)
                                  win = np.hanning(win_len)
                                  han_win = np.zeros(hist_len)
                                  han_win[:win_len] = win
                                  win_center = int(win_len/2)
                                  win_roll = max_hist_ind - win_center
                                  front_pad = 0
                                  back_pad = -0
                                  if win_roll < 0 and max_hist_ind < win_len:
                                      back_pad = win_len - max_hist_ind
                                  elif win_roll > 0 and max_hist_ind > (hist_len - win_len + 1):
                                      front_pad = max_hist_ind - (hist_len - win_len + 1)             
                                  han_win = np.roll(han_win,win_roll)
                                  han_win[:front_pad] = 0
                                  han_win[back_pad:] = 0
                                  han_win_len = len(han_win)
                                  
                                  #self.msg_if.pub_warn([min_range,max_range])
                                  #self.msg_if.pub_warn(bins)
                                  #self.msg_if.pub_warn(han_win)
                                  if np.sum(han_win) > .1:
                                      depth_val=np.average(bins,weights = han_win)
                                  


                          min_filter=depth_val-target_depth_m/2
                          max_filter=depth_val+target_depth_m/2
                          depth_array=depth_array[depth_array > min_filter]
                          depth_array=depth_array[depth_array < max_filter]
                          depth_len=len(depth_array)
                          #self.msg_if.pub_warn("")
                          #self.msg_if.pub_warn(depth_len)
                          if depth_len > target_min_points:
                              target_range_m=depth_val
                          else:
                              target_range_m= -999
                          #self.msg_if.pub_warn(target_range_m)
                          
                      # Calculate target bearings
                      object_loc_y_pix = float(box.ymin + ((box.ymax - box.ymin))  / 2) 
                      object_loc_x_pix = float(box.xmin + ((box.xmax - box.xmin))  / 2)
                      object_loc_y_ratio_from_center = float(object_loc_y_pix - self.img_height/2) / float(self.img_height/2)
                      object_loc_x_ratio_from_center = float(object_loc_x_pix - self.img_width/2) / float(self.img_width/2)
                      target_vert_angle_deg = (object_loc_y_ratio_from_center * float(image_fov_vert/2))
                      target_horz_angle_deg = - (object_loc_x_ratio_from_center * float(image_fov_horz/2))
                      ### Print the range and bearings for each detected object
                      #self.msg_if.pub_warn("")
                      #self.msg_if.pub_warn(target_label)
                      #self.msg_if.pub_warn(str(depth_box_adj.shape) + " detection box size")
                      #self.msg_if.pub_warn("%.2f" % target_range_m + "m : " + "%.2f" % target_horz_angle_deg + "d : " + "%.2f" % target_vert_angle_deg + "d : ")
                      #self.msg_if.pub_warn("")

                      #### Filter targets based on center location and min_px_ratio
                      valid_2d_target = True
                      ref_px_len = math.sqrt(self.img_height**2 + self.img_width**2)
                      px_dist_list = []
                      px_mmx_list = []
                      px_mmy_list = []
                      px_area_list = []
                      for i2, box_class in enumerate(box_class_list):
                          if i2 != i and box.Class == box_class_list[i2]:
                              dif_y = box_center_list[i][0] - box_center_list[i2][0]
                              dif_x = box_center_list[i][1] - box_center_list[i2][1]
                              px_dist = math.sqrt(dif_x**2 + dif_y**2)
                              px_dist_list.append(px_dist)
                              px_mmx_list.append(box_mmx_list[i2])
                              px_mmy_list.append(box_mmy_list[i2])
                              px_area_list.append(box_area_list[i2])


                      for i3, dist in enumerate(px_dist_list): # Check if target is valid
                          if px_area_list[i3] > box_area_list[i]:
                              dist_ratio = dist/ref_px_len
                              if dist_ratio < target_min_px_ratio: # Check if target center is within a bigger box
                                  valid_2d_target = False 
                              if valid_2d_target:
                                  box_x = box_center_list[i][0]
                                  cent_in_x = box_x > box_mmx_list[i3][0] and box_x < box_mmx_list[i3][1]
                                  box_y = box_center_list[i][1]
                                  cent_in_y = box_y > box_mmy_list[i3][0] and box_y < box_mmy_list[i3][1]
                                  if cent_in_x and cent_in_y: # Check if target center is within a bigger box
                                      valid_2d_target = False
                      #self.msg_if.pub_warn("Target Valid: " + target_label + " " + str(valid_2d_target))
                      if valid_2d_target:
                          #### NEED TO Calculate Unique IDs
                          uid_suffix = 0
                          target_uid = box.Class + "_" + str(uid_suffix)# Need to add unque id tracking
                          while target_uid in target_uids:
                              uid_suffix += 1
                              target_uid = box.Class + "_" + str(uid_suffix)
                          target_uids.append(target_uid)
                          bounding_box_3d_msg = None

                          #self.msg_if.pub_warn("Target Selected: " + str(self.selected_target))
                          #self.msg_if.pub_warn("Target Uid: " + str(target_uid))
                          if self.selected_target == "None" or self.selected_target == target_uid:
                              # Updated Bounding Box 2d
                              bounding_box_msg = BoundingBox()
                              bounding_box_msg.Class = box.Class
                              bounding_box_msg.id = box.id
                              bounding_box_msg.uid = target_uid
                              bounding_box_msg.probability = box.probability
                              bounding_box_msg.xmin = box.xmin
                              bounding_box_msg.xmax = box.xmax
                              bounding_box_msg.ymin = box.ymin
                              bounding_box_msg.ymax = box.ymax
                              bounding_box_msg.area_pixels = area_pixels
                              bounding_box_msg.area_ratio = area_ratio
                              bbs2d.append(bounding_box_msg)

                              # Create Bounding Box 3d
                              area_meters = -999
                              volume_meters = -999
                              if target_range_m != -999:
                                  target_depth = selected_classes_dict[box.Class]['depth']
                                  # Calculate Bounding Box 3D
                                  bounding_box_3d_msg = BoundingBox3D()
                                  bounding_box_3d_msg.Class = box.Class
                                  bounding_box_3d_msg.id = box.id 
                                  bounding_box_3d_msg.uid = target_uid
                                  bounding_box_3d_msg.probability = box.probability
                                  # Calculate the Box Center
                                  # Ref www.stackoverflow.com/questions/30619901/calculate-3d-point-coordinates-using-horizontal-and-vertical-angles-and-slope-di
                                  bbc = Vector3()
                                  theta_deg = (target_vert_angle_deg + 90)  #  Vert Angle 0 - 180 from top
                                  theta_rad = theta_deg * math.pi/180 #  Vert Angle 0 - PI from top
                                  phi_deg =  (target_horz_angle_deg) # Horz Angle 0 - 360 from X axis counter clockwise
                                  phi_rad = phi_deg * math.pi/180 # Horz Angle 0 - 2 PI from from X axis counter clockwise
                                
                                  bbc.x = target_range_m * math.sin(theta_rad) * math.cos(phi_rad) - transform[0]
                                  bbc.y = target_range_m * math.sin(theta_rad) * math.sin(phi_rad) - transform[1]
                                  bbc.z = target_range_m * math.cos(theta_rad) - transform[2]
                                  #self.msg_if.pub_warn([target_range_m,theta_deg,phi_deg,bbc.x, bbc.y,bbc.z])
                                  bounding_box_3d_msg.box_center_m.x = bbc.x + target_depth / 2
                                  bounding_box_3d_msg.box_center_m.y = bbc.y
                                  bounding_box_3d_msg.box_center_m.z = bbc.z 

                                  # Calculate the Box Extent
                                  bbe = Vector3()  
                                  mpp_vert_at_range = 2 * target_range_m * math.sin(image_fov_vert/2 * math.pi/180) / self.img_height
                                  mpp_horz_at_range = 2* target_range_m * math.sin(image_fov_horz/2 * math.pi/180) / self.img_width
                                  mpp_at_range = (mpp_vert_at_range + mpp_horz_at_range) / 2  #  ToDo: More accurate calc
                                  bbe.x = target_depth
                                  bbe.y = mpp_at_range * (box.xmax-box.xmin)
                                  bbe.z = mpp_at_range * (box.ymax-box.ymin)
                                  bounding_box_3d_msg.box_extent_xyz_m.x = bbe.x
                                  bounding_box_3d_msg.box_extent_xyz_m.y = bbe.y
                                  bounding_box_3d_msg.box_extent_xyz_m.z = bbe.z
                                  # Target Rotation (roll,pitch,yaw)
                                  bbr = Vector3()
                                  bbr.x = -transform[3]
                                  bbr.y = -transform[4]
                                  bbr.z = -transform[5]
                                  bounding_box_3d_msg.box_rotation_rpy_deg.x = bbr.x
                                  bounding_box_3d_msg.box_rotation_rpy_deg.y = bbr.y
                                  bounding_box_3d_msg.box_rotation_rpy_deg.z = bbr.z
                                  # To Do Add Bounding Box 3D Data

                                  area_meters = bbe.y * bbe.z
                                  volume_meters = area_meters * bbe.x
                                  bounding_box_3d_msg.volume_meters = volume_meters
                                  bbs3d.append(bounding_box_3d_msg)

                                  # Now update range and bearing values based on transform
                                  target_range_m = math.sqrt(bbc.x**2 + bbc.y**2 + bbc.z**2)
                                  #self.msg_if.pub_warn(str([bbc.x,bbc.y,bbc.z]))


                                  try:
                                    horz_ang = np.arctan(bbc.x/(bbc.y)) * 180/math.pi
                                    target_horz_angle_deg = np.sign(horz_ang) * (90 - abs(horz_ang)) - transform[5]
                                  except:
                                    target_horz_angle_deg = -999
                                  try:
                                    vert_ang = np.arctan(bbc.x/(-bbc.z)) * 180/math.pi
                                    target_vert_angle_deg = np.sign(vert_ang) * (90 - abs(vert_ang)) - transform[4]
                                  except:
                                    target_vert_angle_deg = -999

                              # Create target_localizations
                              target_data_msg=TargetLocalization()
                              target_data_msg.Class = box.Class
                              target_data_msg.id = box.id 
                              target_data_msg.uid = target_uid
                              target_data_msg.xmin = box.xmin
                              target_data_msg.xmax = box.xmax
                              target_data_msg.ymin = box.ymin
                              target_data_msg.ymax = box.ymax

                              target_data_msg.range_m=target_range_m
                              target_data_msg.azimuth_deg=target_horz_angle_deg
                              target_data_msg.elevation_deg=target_vert_angle_deg
                              target_data_msg.area_pixels = area_pixels
                              target_data_msg.area_ratio = area_ratio
                              target_data_msg.area_meters = area_meters
                              tls.append(target_data_msg)

                              # Update Current Targets List
                              if bounding_box_3d_msg is not None:
                                  center_m = [bbc.x,bbc.y,bbc.z]
                              else:
                                  center_m = [-999,-999,-999]

                              current_targets_dict[target_uid] = {
                                  'image_seq_num': image_seq_num,
                                  'class_name': box.Class, 
                                  'target_uid': target_uid,
                                  'bounding_box': [box.xmin,box.xmax,box.ymin,box.ymax],
                                  'bounding_box_adj': [xmin_adj,xmax_adj,ymin_adj,ymax_adj],
                                  'range_bearings': [target_range_m , target_horz_angle_deg , target_vert_angle_deg],
                                  'center_px': [box.xmax-box.xmin,box.ymax-box.ymin],
                                  'velocity_pxps': [0,0],
                                  'center_m': center_m,
                                  'area_pixels': area_pixels,
                                  'area_ratio': area_ratio,
                                  'area_meters': area_meters,
                                  'volume_meters': volume_meters,
                                  'velocity_mps': [0,0,0],
                                  'last_detection_timestamp': get_msg_timestamp                              
                                  }
                              active_targets_dict[target_uid] = current_targets_dict[target_uid]

      #self.msg_if.pub_warn("Created active targets dict: " + str(active_targets_dict))
      self.bbs_msg = bbs_msg
      self.active_targets_dict = active_targets_dict
      if current_targets_dict.keys() != self.current_targets_dict.keys():
          self.publish_targets()
      self.current_targets_dict = current_targets_dict
      #self.msg_if.pub_warn(self.current_targets_dict)
      # Publish and Save 2D Bounding Boxes
      if len(bbs2d) > 0:
        bbs_msg.bounding_boxes = bbs2d
        if not nepi_sdk.is_shutdown():

          if self.node_if is not None:
            self.node_if.publish_pub('target_boxes_2d_pub', bbs_msg)
          oc_msg = ObjectCount()
          oc_msg.header = detect_header
          oc_msg.count = len(bbs_msg.bounding_boxes)
          if self.node_if is not None:
            self.node_if.publish_pub('box_count_pub', oc_msg)
        # Save Data if it is time.
        bbs_dict = dict()
        bbs_dict['timestamp'] =  nepi_sdk.get_datetime_str_from_stamp(bbs_msg.header.stamp)
        bbs_dict['image_topic'] = bbs_msg.image_topic
        bbs_dict['image_height'] = bbs_msg.image_height
        bbs_dict['image_width'] = bbs_msg.image_width
        bb_list = []
        for ind, bb_msg in enumerate(bbs_msg.bounding_boxes):
            bb_dict = dict()
            bb_dict['class'] = bb_msg.Class
            bb_dict['id'] = bb_msg.id
            bb_dict['uid'] = bb_msg.uid
            bb_dict['probability'] = bb_msg.probability
            bb_dict['xmin'] = bb_msg.xmin
            bb_dict['ymin'] = bb_msg.ymin
            bb_dict['xmax'] = bb_msg.xmax
            bb_dict['ymax'] = bb_msg.ymax
            bb_dict['area_pixels'] = bb_msg.area_pixels
            bb_dict['area_ratio'] = bb_msg.area_ratio
            bb_list.append(bb_dict)
        bbs_dict['bounding_boxes'] = bb_list
        self.save_data_if.save_dict2file("target_boxes_2d",bbs_dict,get_msg_timestamp)

      # Publish and Save Target Localizations
      #self.msg_if.pub_warn("Got tls list: " + str(tls))
      self.target_locs_lock.acquire()
      self.target_locs = tls      
      self.target_locs_lock.release()

      if len(tls) > 0:
        tls_msg = TargetLocalizations()
        tls_msg.header = detect_header
        tls_msg.image_topic = self.current_image_topic
        tls_msg.image_header = self.current_image_header
        tls_msg.image_height = bbs_msg.image_height
        tls_msg.image_width = bbs_msg.image_width
        tls_msg.depth_topic = self.depth_map_topic
        tls_msg.depth_header = self.depth_map_header
        tls_msg.target_localizations = tls

        #self.msg_if.pub_warn("Will pub tls msg: " + str(tls_msg))
        if self.node_if is not None:
          self.node_if.publish_pub('target_localizations_pub', tls_msg)

        # Save Data if Time
        tls_dict = dict()
        tls_dict['timestamp'] =  nepi_sdk.get_datetime_str_from_stamp(tls_msg.header.stamp)
        tls_dict['image_topic'] = tls_msg.image_topic
        tls_dict['image_height'] = tls_msg.image_height
        tls_dict['image_width'] = tls_msg.image_width
        tls_dict['depth_topic'] = tls_msg.depth_topic
        tl_list = []
        for ind, tl_msg in enumerate(tls_msg.target_localizations):
            tl_dict = dict()
            tl_dict['class'] = tl_msg.Class
            tl_dict['id'] = tl_msg.id
            tl_dict['uid'] = tl_msg.uid
            tl_dict['confidence'] = tl_msg.confidence
            tl_dict['range_m'] = tl_msg.range_m
            tl_dict['azimuth_deg'] = tl_msg.azimuth_deg
            tl_dict['elevation_deg'] = tl_msg.elevation_deg
            tl_dict['covariance'] = tl_msg.position_covariance
            tl_dict['area_pixels'] = tl_msg.area_pixels
            tl_dict['area_ratio'] = tl_msg.area_ratio
            tl_list.append(tl_dict)
        tls_dict['target_locs'] = tl_list
        self.save_data_if.save_dict2file('target_localizations',tls_dict,get_msg_timestamp)

      # Pub target count
      tc_msg = ObjectCount()
      tc_msg.header = detect_header
      tc_msg.count = len(tls)
      if self.node_if is not None:
        self.node_if.publish_pub('target_count_pub', tc_msg)


      # Publish and Save 3D Bounding Boxes
      self.target_box_3d_list = bbs3d
      #self.msg_if.pub_warn("")
      #self.msg_if.pub_warn(bbs3d)
      if len(bbs3d) > 0:
        bb3s_msg = BoundingBoxes3D()
        bb3s_msg.header = detect_header
        bb3s_msg.image_topic = self.current_image_topic
        bb3s_msg.image_header = self.current_image_header
        bb3s_msg.image_height = bbs_msg.image_height
        bb3s_msg.image_width = bbs_msg.image_width
        bb3s_msg.depth_map_header = self.depth_map_header
        bb3s_msg.depth_map_topic = self.depth_map_topic
        bb3s_msg.bounding_boxes_3d = bbs3d
        if self.node_if is not None:
          self.node_if.publish_pub('target_boxes_3d_pub', bb3s_msg)
          oc3_msg = ObjectCount()
          oc3_msg.header = detect_header
          oc3_msg.count = len(bbs3d)
          self.node_if.publish_pub('box3d_count_pub', oc3_msg)

        # Save Data if Time
        bb3s_dict = dict()
        bb3s_dict['timestamp'] =  nepi_sdk.get_datetime_str_from_stamp(bb3s_msg.header.stamp)
        bb3s_dict['image_topic'] = bb3s_msg.image_topic
        bb3s_dict['image_height'] = bb3s_msg.image_height
        bb3s_dict['image_width'] = bb3s_msg.image_width
        bb3s_dict['depth_map_topic'] = bb3s_msg.depth_map_topic
        bb3_list = []
        for ind, bb3_msg in enumerate(bb3s_msg.bounding_boxes_3d):
            bb3_dict = dict()
            bb3_dict['class'] = bb3_msg.Class
            bb3_dict['id'] = bb3_msg.id
            bb3_dict['uid'] = bb3_msg.uid
            bb3_dict['probability'] = bb3_msg.probability
            bb3_dict['box_center_m'] = bb3_msg.box_center_m
            bb3_dict['box_extent_xyz_m'] = bb3_msg.box_extent_xyz_m
            bb3_dict['box_rotation_rpy_deg'] = bb3_msg.box_rotation_rpy_deg
            bb3_dict['volume_meters'] = bb3_msg.volume_meters
            bb3_list.append(bb3_dict)
        bb3s_dict['bounding_boxes_3d'] = bb3_list
        self.save_data_if.save_dict2file('target_boxes_3d',bb3s_dict,get_msg_timestamp)


  def imagePubCb(self,timer):
    data_product = 'targeting_image'
    app_enabled = self.app_enabled
    if app_enabled and self.image_sub is not None and self.classifier_running and self.classes_selected:
      has_subscribers = self.image_if.has_subscribers_check()
      #self.msg_if.pub_warn("Checking for subscribers: " + str(has_subscribers))
      saving_is_enabled = self.save_data_if.data_product_saving_enabled(data_product)
      snapshot_enabled = self.save_data_if.data_product_snapshot_enabled(data_product)
      should_save = (saving_is_enabled and self.save_data_if.data_product_should_save(data_product)) or snapshot_enabled
      #self.msg_if.pub_warn("Checking for save_: " + str(should_save))

      if has_subscribers or should_save:
        self.img_lock.acquire()
        img_msg = copy.deepcopy(self.img_msg)
        self.img_msg = None
        self.img_lock.release()
        if img_msg is not None:
          self.target_locs_lock.acquire()
          tls = self.target_locs      
          self.target_locs_lock.release()

          current_image_header = img_msg.header
          get_msg_timestamp = img_msg.header.stamp     
          cv2_img = nepi_img.rosimg_to_cv2img(img_msg).astype(np.uint8)
          cv2_shape = cv2_img.shape
          self.img_width = cv2_shape[1] 
          self.img_height = cv2_shape[0]   
        
          for target_loc in tls:
            class_name = target_loc.Class
            target_uid = target_loc.uid
            target_range_m = target_loc.range_m
            target_horz_angle_deg = target_loc.azimuth_deg
            target_vert_angle_deg = target_loc.elevation_deg
            ###### Apply Image Overlays and Publish Targeting_Image ROS Message
            # Overlay adjusted detection boxes on image 
            [xmin,xmax,ymin,ymax] = [target_loc.xmin,target_loc.xmax,target_loc.ymin,target_loc.ymax]
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            class_name = class_name

            class_color = (255,0,0)
            if class_name in self.classes_list:
                class_ind = self.classes_list.index(class_name)
                if class_ind < len(self.class_color_list):
                    class_color = tuple(self.class_color_list[class_ind])
            line_thickness = 2
            cv2.rectangle(cv2_img, start_point, end_point, class_color, thickness=line_thickness)
            # Overlay text data on OpenCV image
            font                   = cv2.FONT_HERSHEY_DUPLEX
            fontScale, thickness  = nepi_img.optimal_font_dims(cv2_img,font_scale = 1.5e-3, thickness_scale = 1.5e-3)
            fontColor = (255, 255, 255)
            lineType = 1

            ## Overlay Label
            text2overlay=target_uid
            text_size = cv2.getTextSize(text2overlay, 
                font, 
                fontScale,
                thickness)
            #self.msg_if.pub_warn("Text Size: " + str(text_size))
            line_height = text_size[0][1]
            line_width = text_size[0][0]
            bottomLeftCornerOfText = (xmin + line_thickness,ymin + line_thickness * 2 + line_height)
            # Create Text Background Box
            padding = int(line_height*0.4)
            start_point = (bottomLeftCornerOfText[0]-padding, bottomLeftCornerOfText[1]-line_height-padding)
            end_point = (bottomLeftCornerOfText[0]+line_width+padding, bottomLeftCornerOfText[1]+padding)
            box_color = [0,0,0]
            cv2.rectangle(cv2_img, start_point, end_point, box_color , -1)

            cv2.putText(cv2_img,text2overlay, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
          
            # Overlay Data
            #self.msg_if.pub_warn(line_height)
            if target_range_m == -999:
              tr = '#'
            else:
              tr = ("%.1f" % target_range_m )
            if target_horz_angle_deg == -999:
              th = '#'
            else:
              th = ("%.f" % target_horz_angle_deg)

            if target_vert_angle_deg == -999:
              tv = '#'
            else:
              tv = ("%.f" % target_vert_angle_deg)

            text2overlay= tr + "m," + th + "d," + tv + "d"
            text_size = cv2.getTextSize(text2overlay, 
                font, 
                fontScale,
                thickness)
            line_height = text_size[0][1]
            line_width = text_size[0][0]
            bottomLeftCornerOfText = (xmin + line_thickness,ymin + line_thickness * 2 + line_height * 3)
            # Create Text Background Box
            padding = int(line_height*0.4)
            start_point = (bottomLeftCornerOfText[0]-padding, bottomLeftCornerOfText[1]-line_height-padding)
            end_point = (bottomLeftCornerOfText[0]+line_width+padding, bottomLeftCornerOfText[1]+padding)
            box_color = [0,0,0]
            cv2.rectangle(cv2_img, start_point, end_point, box_color , -1)
            cv2.putText(cv2_img,text2overlay, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType) 

          # Publish new image to ros
          if not nepi_sdk.is_shutdown() and has_subscribers: #and has_subscribers:
              #Convert OpenCV image to ROS image
              cv2_shape = cv2_img.shape
              if  cv2_shape[2] == 3:
                encode = 'bgr8'
              else:
                encode = 'mono8'
              self.image_if.publish_cv2_image(cv2_img, timestamp = get_msg_timestamp, encoding = encode)
          # Save Data if Time
          if should_save:
            self.save_data_if.save_img2file(data_product,cv2_img,get_msg_timestamp,save_check = False)




  def imageCb(self,img_msg):    
      self.img_lock.acquire()
      self.img_msg = img_msg
      self.img_lock.release()



  ### Monitor Output of AI model to clear detection status
  def foundObjectCb(self,found_obj_msg):
    app_enabled = self.app_enabled

    #Clean Up
    if found_obj_msg.count != 0:
      self.no_object_count = 0
    else:
      #print("No objects detected")
      self.no_object_count += 1
      self.target_locs_lock.acquire()
      self.target_locs = []     
      self.target_locs_lock.release()
      
      self.bbs_msg = None
      self.target_box_3d_list = None
      self.current_targets_dict = dict()
      tc_msg = ObjectCount()
      tc_msg.header = found_obj_msg.header
      tc_msg.count = 0
      if self.node_if is not None:
        self.node_if.publish_pub('target_count_pub', tc_msg)
        self.node_if.publish_pub('box3d_count_pub', tc_msg)


  def depthMapCb(self,depth_map_msg):
    self.current_detph_map_header = depth_map_msg.header
    # Zed depth data is floats in m, but passed as 4 bytes each that must be converted to floats
    # Use cv2_bridge() to convert the ROS image to OpenCV format
    #Convert the depth 4xbyte data to global float meter array
    self.depth_map_header = depth_map_msg.header
    cv2_depth_image = self.cv2_bridge.imgmsg_to_cv2(depth_map_msg, desired_encoding="passthrough")
    #cv2_depth_image = nepi_img.rosimg_to_cv2img(depth_map_msg)
    self.np_depth_array_m = (np.array(cv2_depth_image, dtype=np.float32)) # replace nan values
    self.np_depth_array_m[np.isnan(self.np_depth_array_m)] = 0 # zero pixels with no value
    self.np_depth_array_m[np.isinf(self.np_depth_array_m)] = 0 # zero pixels with inf value


                
    
  #######################
  # Node Cleanup Function
  
  def cleanup_actions(self):
    self.msg_if.pub_info(" Shutting down: Executing script cleanup actions")


#########################################
# Main
#########################################
if __name__ == '__main__':
  NepiAiTargetingApp()