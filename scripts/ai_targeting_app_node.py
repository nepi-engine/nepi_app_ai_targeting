#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.
#
# This file is part of nepi-engine
# (see https://github.com/nepi-engine).
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#


import os
# ROS namespace setup
#NEPI_BASE_NAMESPACE = '/nepi/s2x/'
#os.environ["ROS_NAMESPACE"] = NEPI_BASE_NAMESPACE[0:-1] # remove to run as automation script
import rospy



import time
import sys
import numpy as np
import cv2
import copy
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nepi_edge_sdk_base import nepi_ros
from nepi_edge_sdk_base import nepi_pc 
from nepi_edge_sdk_base import nepi_img 

from std_msgs.msg import UInt8, Int32, Float32, Empty, String, Bool, Header
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from cv_bridge import CvBridge
from nepi_ros_interfaces.msg import ClassifierSelection,  \
                                    BoundingBox, BoundingBoxes, BoundingBox3D, BoundingBoxes3D, \
                                    ObjectCount, ClassifierSelection, \
                                    AiTargetingStatus, \
                                    StringArray, TargetLocalization, TargetLocalizations
from nepi_ros_interfaces.srv import ImageClassifierStatusQuery, ImageClassifierStatusQueryRequest
from nepi_ros_interfaces.msg import Frame3DTransform
from nepi_edge_sdk_base.save_data_if import SaveDataIF
from nepi_edge_sdk_base.save_cfg_if import SaveCfgIF

# Do this at the end
#from scipy.signal import find_peaks


NEPI_BASE_NAMESPACE = nepi_ros.get_base_namespace()

#########################################

#########################################
# Node Class
#########################################

class NepiAiTargetingApp(object):

  #Set Initial Values
  FACTORY_FOV_VERT_DEG=70 # Camera Vertical Field of View (FOV)
  FACTORY_FOV_HORZ_DEG=110 # Camera Horizontal Field of View (FOV)
  FACTORY_TARGET_BOX_SIZE_PERCENT=80 # percent ajustment on detection box around center to use for range calc
  FACTORY_TARGET_DEPTH_METERS=0.5 # Sets the depth filter around mean depth to use for range calc
  FACTORY_TARGET_MIN_POINTS=10 # Sets the minimum number of valid points to consider for a valid range
  FACTORY_TARGET_MIN_PX_RATIO=0.1 # Sets the minimum px range between two detections, largest box between two is selected
  FACTORY_TARGET_MIN_DIST_METERS=0.0 # Sets the minimum distance between to detections, closer target is selected
  FACTORY_TARGET_MAX_AGE_SEC=10 # Remove lost targets from dictionary if older than this age

  FACTORY_OUTPUT_IMAGE = "targeting_image"

  ZERO_TRANSFORM = [0,0,0,0,0,0,0]

  NONE_CLASSES_DICT = dict()
  NONE_CLASSES_DICT["None"] = {'depth': FACTORY_TARGET_DEPTH_METERS}

  EMPTY_TARGET_DICT = dict()
  EMPTY_TARGET_DICT["None"] = {     'class_name': 'None', 
                                    'target_uid': 'None',
                                    'bounding_box': [0,0,0,0],
                                    'range_bearings': [0,0,0],
                                    'center_px': [0,0],
                                    'velocity_pxps': [0,0],
                                    'enter_m': [0,0,0],
                                    'velocity_mps': [0,0,0],
                                    'last_detection_timestamp': rospy.Duration(0)                              
                                    }


  targeting_running = False
  data_products = ["image",'targeting_image_depth',"detection_boxes","targeting_boxes_2d","targeting_boxes_3d","targeting_localizations"]
  output_image_options = ["detection_image","alert_image","targeting_image"]
  
  current_classifier = "None"
  current_classifier_state = "None"
  classes_list = []
  current_classifier_classes = "[]"


  cv2_bridge = CvBridge()
  current_image_topic = "None"
  current_image_header = Header()
  image_source_topic = ""
  img_width = 0
  img_height = 0
  image_sub = None
  last_img_msg = None

  depth_map_topic = "None"
  depth_map_header = Header()
  depth_map_sub = None
  np_depth_array_m = None
  has_depth_map = False
  pointcloud_topic = "None"
  has_pointcloud = False

  detect_boxes_msg = None
  targeting_box_3d_list = None
  class_color_list = []

  last_targeting_enable = False
  last_image_topic = "None"
  
  selected_classes_dict = dict()
  current_targets_dict = dict()
  active_targets_dict = dict()
  lost_targets_dict = dict()
  selected_target = "All"

  targeting_class_list = []
  targeting_target_list = []
  alert_image_pub = None
  alert_status_pub = None
  alert_status = False
  detection_image_pub = None
  targeting_image_pub = None


  seq = 0

  #has_subscribers_detect_img = False
  has_subscribers_target_img = False

  #######################
  ### App Config Functions

  def resetAppCb(self,msg):
    self.resetApp()

  def resetApp(self):
    rospy.set_param('~last_classifier', "")
    rospy.set_param('~selected_output_image', self.FACTORY_OUTPUT_IMAGE)
    rospy.set_param('~selected_classes_dict', dict())
    rospy.set_param('~image_fov_vert',  self.FACTORY_FOV_VERT_DEG)
    rospy.set_param('~image_fov_horz', self.FACTORY_FOV_HORZ_DEG)
    rospy.set_param('~target_box_percent',  self.FACTORY_TARGET_BOX_SIZE_PERCENT)
    rospy.set_param('~default_target_depth',  self.FACTORY_TARGET_DEPTH_METERS)
    rospy.set_param('~target_min_points', self.FACTORY_TARGET_MIN_POINTS)
    rospy.set_param('~target_min_px_ratio', self.FACTORY_TARGET_MIN_PX_RATIO)
    rospy.set_param('~target_min_dist_m', self.FACTORY_TARGET_MIN_DIST_METERS)
    rospy.set_param('~target_age_filter', self.FACTORY_TARGET_MAX_AGE_SEC)
    rospy.set_param('~frame_3d_transform', self.ZERO_TRANSFORM)
    self.current_targets_dict = dict()
    self.lost_targets_dict = dict()
    self.publish_status()

  def saveConfigCb(self, msg):  # Just update Class init values. Saving done by Config IF system
    pass # Left empty for sim, Should update from param server

  def setCurrentAsDefault(self):
    self.initParamServerValues(do_updates = False)

  def updateFromParamServer(self):
    #rospy.logwarn("Debugging: param_dict = " + str(param_dict))
    #Run any functions that need updating on value change
    # Don't need to run any additional functions
    pass

  def initParamServerValues(self,do_updates = True):
      rospy.loginfo("AI_TRG_APP: Setting init values to param values")
      self.init_last_classifier = rospy.get_param("~last_classifier", "")
      self.init_selected_output_image = rospy.get_param('~selected_output_image', self.FACTORY_OUTPUT_IMAGE)
      self.init_selected_classes_dict = rospy.get_param('~selected_classes_dict', dict())
      self.init_image_fov_vert = rospy.get_param('~image_fov_vert',  self.FACTORY_FOV_VERT_DEG)
      self.init_image_fov_horz = rospy.get_param('~image_fov_horz', self.FACTORY_FOV_HORZ_DEG)
      self.init_target_box_adjust = rospy.get_param('~target_box_percent',  self.FACTORY_TARGET_BOX_SIZE_PERCENT)
      self.init_default_target_depth = rospy.get_param('~default_target_depth',  self.FACTORY_TARGET_DEPTH_METERS)
      self.init_target_min_points = rospy.get_param('~target_min_points', self.FACTORY_TARGET_MIN_POINTS)
      self.init_target_min_px_ratio = rospy.get_param('~target_min_px_ratio', self.FACTORY_TARGET_MIN_PX_RATIO)
      self.init_target_min_dist_m = rospy.get_param('~target_min_dist_m', self.FACTORY_TARGET_MIN_DIST_METERS)
      self.init_target_age_filter = rospy.get_param('~target_age_filter', self.FACTORY_TARGET_MAX_AGE_SEC)
      self.init_frame_3d_transform = rospy.get_param('~frame_3d_transform', self.ZERO_TRANSFORM)
      self.resetParamServer(do_updates)

  def resetParamServer(self,do_updates = True):
      rospy.set_param('~last_classiier', self.init_last_classifier)
      rospy.set_param('~selected_output_image', self.init_selected_output_image)
      rospy.set_param('~selected_classes_dict', self.init_selected_classes_dict)
      rospy.set_param('~image_fov_vert',  self.init_image_fov_vert)
      rospy.set_param('~image_fov_horz', self.init_image_fov_horz)
      rospy.set_param('~target_box_percent',  self.init_target_box_adjust)
      rospy.set_param('~default_target_depth',  self.init_default_target_depth)
      rospy.set_param('~target_min_points', self.init_target_min_points)
      rospy.set_param('~target_min_px_ratio', self.init_target_min_px_ratio)
      rospy.get_param('~target_min_dist_m', self.init_target_min_dist_m)
      rospy.set_param('~target_age_filter', self.init_target_age_filter)
      rospy.set_param('~frame_3d_transform', self.init_frame_3d_transform)
      if do_updates:
          self.updateFromParamServer()
          self.publish_status()



  ###################
  ## AI Manager Passthrough Callbacks

  def startClassifierCb(self,msg):
    ##rospy.loginfo(msg)
    self.start_classifier_pub.publish(msg)

  def stopClassifierCb(self,msg):
    ##rospy.loginfo(msg)
    self.stop_classifier_pub.publish(msg)

  def setThresholdCb(self,msg):
    ##rospy.loginfo(msg)
    self.set_threshold_pub.publish(msg)

  ###################
  ## AI App Callbacks


  def selectImageCb(self,msg):
    ##rospy.loginfo(msg)
    image_name = msg.data
    if image_name in self.output_image_options:
      rospy.set_param('~selected_output_image', image_name)
    self.publish_status()


  def addAllClassesCb(self,msg):
    ##rospy.loginfo(msg)
    classes = self.current_classifier_classes
    depth = rospy.get_param('~default_target_depth',self.init_default_target_depth)
    selected_dict = dict()
    for Class in classes:
      selected_dict[Class] = {'depth': depth }
    rospy.set_param('~selected_classes_dict', selected_dict)
    self.publish_status()

  def removeAllClassesCb(self,msg):
    ##rospy.loginfo(msg)
    rospy.set_param('~selected_classes_dict', dict())
    self.publish_status()

  def addClassCb(self,msg):
    ##rospy.loginfo(msg)
    class_name = msg.data
    class_depth_m = rospy.get_param('~default_target_depth',  self.init_default_target_depth)
    if class_name in self.current_classifier_classes:
      selected_classes_dict = rospy.get_param('~selected_classes_dict', self.init_selected_classes_dict)
      selected_classes_dict[class_name] = {'depth': class_depth_m}
      rospy.set_param('~selected_classes_dict', selected_classes_dict)
    self.publish_status()

  def removeClassCb(self,msg):
    ##rospy.loginfo(msg)
    class_name = msg.data
    selected_classes_dict = rospy.get_param('~selected_classes_dict', self.init_selected_classes_dict)
    if class_name in selected_classes_dict.keys():
      del selected_classes_dict[class_name]
      rospy.set_param('~selected_classes_dict', selected_classes_dict)
    self.publish_status()


  def selectTargetCb(self,msg):
    ##rospy.loginfo(msg)
    target_name = msg.data
    if target_name == 'All' or target_name in self.current_targets_dict.keys():
      self.selected_target = target_name
    self.publish_status()

  def setVertFovCb(self,msg):
    ##rospy.loginfo(msg)
    fov = msg.data
    if fov > 0:
      rospy.set_param('~image_fov_vert',  fov)
    self.publish_status()


  def setHorzFovCb(self,msg):
    ##rospy.loginfo(msg)
    fov = msg.data
    if fov > 0:
      rospy.set_param('~image_fov_horz',  fov)
    self.publish_status()
    
  def setTargetBoxPercentCb(self,msg):
    #rospy.loginfo(msg)
    val = msg.data
    if val >= 10 and val <= 200:
      rospy.set_param('~target_box_percent',val)
    self.publish_status()   
      
  def setDefaultTargetDepthCb(self,msg):
    #rospy.loginfo(msg)
    val = msg.data
    if val >= 0:
      rospy.set_param('~default_target_depth',val)
    self.publish_status()   

  def setTargetMinPointsCb(self,msg):
    #rospy.loginfo(msg)
    val = msg.data
    if val >= 0:
      rospy.set_param('~target_min_points',val)
    self.publish_status() 

  def setTargetMinPxRatioCb(self,msg):
    #rospy.loginfo(msg)
    val = msg.data
    if val >= 0 and val <= 1:
      rospy.set_param('~target_min_px_ratio',val)
    self.publish_status() 

  def setTargetMinDistMCb(self,msg):
    #rospy.loginfo(msg)
    val = msg.data
    if val >= 0:
      rospy.set_param('~target_min_dist_m',val)
    self.publish_status() 

  def setAgeFilterCb(self,msg):
    #rospy.loginfo(msg)
    val = msg.data
    if val >= 0:
      rospy.set_param('~target_age_filter',val)
    self.publish_status()

  def setFrame3dTransformCb(self, msg):
      new_transform_msg = msg
      self.setFrame3dTransform(new_transform_msg)

  def setFrame3dTransform(self, transform_msg):
      x = transform_msg.translate_vector.x
      y = transform_msg.translate_vector.y
      z = transform_msg.translate_vector.z
      roll = transform_msg.rotate_vector.x
      pitch = transform_msg.rotate_vector.y
      yaw = transform_msg.rotate_vector.z
      heading = transform_msg.heading_offset
      transform = [x,y,z,roll,pitch,yaw,heading]
      self.init_frame_3d_transform = rospy.set_param('~frame_3d_transform',  transform)
      self.status_msg.frame_3d_transform = transform_msg
      self.publishStatus(do_updates=False) # Updated inline here 


  #######################
  ### Node Initialization
  def __init__(self):
   
    rospy.loginfo("AI_TRG_APP: Starting Initialization Processes")
    self.initParamServerValues(do_updates = False)
    self.resetParamServer(do_updates = False)
   

    # Set up save data and save config services ########################################################
    factory_data_rates= {}
    for d in self.data_products:
        factory_data_rates[d] = [0.0, 0.0, 100.0] # Default to 0Hz save rate, set last save = 0.0, max rate = 100.0Hz
    if 'detection_image' in self.data_products:
        factory_data_rates['detection_image'] = [1.0, 0.0, 100.0] 
    self.save_data_if = SaveDataIF(data_product_names = self.data_products, factory_data_rate_dict = factory_data_rates)
    # Temp Fix until added as NEPI ROS Node
    self.save_cfg_if = SaveCfgIF(updateParamsCallback=self.initParamServerValues, 
                                 paramsModifiedCallback=self.updateFromParamServer)


    ## App Setup ########################################################
    app_reset_app_sub = rospy.Subscriber('~reset_app', Empty, self.resetAppCb, queue_size = 10)
    self.initParamServerValues(do_updates=False)

    # AI Management Scubscirbers and publishers
  
    rospy.Subscriber('~start_classifier', ClassifierSelection, self.startClassifierCb)
    self.start_classifier_pub = rospy.Publisher(NEPI_BASE_NAMESPACE + "ai_detector_mgr/start_classifier", ClassifierSelection, queue_size=1)
    rospy.Subscriber('~stop_classifier', Empty, self.stopClassifierCb)
    self.stop_classifier_pub = rospy.Publisher(NEPI_BASE_NAMESPACE + "ai_detector_mgr/stop_classifier", Empty, queue_size=1)
    rospy.Subscriber('~set_threshold', Float32, self.setThresholdCb) 
    self.set_threshold_pub = rospy.Publisher(NEPI_BASE_NAMESPACE + "ai_detector_mgr/set_threshold", Float32, queue_size=1)

    # App Specific Subscribers
    sel_image_sub = rospy.Subscriber('~set_output_image', String, self.selectImageCb, queue_size = 10)
    add_all_sub = rospy.Subscriber('~add_all_target_classes', Empty, self.addAllClassesCb, queue_size = 10)
    remove_all_sub = rospy.Subscriber('~remove_all_target_classes', Empty, self.removeAllClassesCb, queue_size = 10)
    add_class_sub = rospy.Subscriber('~add_target_class', String, self.addClassCb, queue_size = 10)
    remove_class_sub = rospy.Subscriber('~remove_target_class', String, self.removeClassCb, queue_size = 10)
    select_target_sub = rospy.Subscriber('~select_target', String, self.selectTargetCb, queue_size = 10)
    vert_fov_sub = rospy.Subscriber("~set_image_fov_vert", Float32, self.setVertFovCb, queue_size = 10)
    horz_fov_sub = rospy.Subscriber("~set_image_fov_horz", Float32, self.setHorzFovCb, queue_size = 10)
    target_box_size_sub = rospy.Subscriber("~set_target_box_size_percent", Int32, self.setTargetBoxPercentCb, queue_size = 10)
    default_target_depth_sub = rospy.Subscriber("~set_default_target_detpth", Float32, self.setDefaultTargetDepthCb, queue_size = 10)
    target_min_points_sub = rospy.Subscriber("~set_target_min_points", Int32, self.setTargetMinPointsCb, queue_size = 10)
    target_min_px_ratio_sub = rospy.Subscriber("~set_target_min_px_ratio", Float32, self.setTargetMinPxRatioCb, queue_size = 10)
    age_filter_sub = rospy.Subscriber("~set_age_filter", Float32, self.setAgeFilterCb, queue_size = 10)
    rospy.Subscriber('~set_3d_transform', Frame3DTransform, self.setFrame3dTransformCb, queue_size=1)

    # Start an AI manager status monitoring thread
    AI_MGR_STATUS_SERVICE_NAME = NEPI_BASE_NAMESPACE + "ai_detector_mgr/img_classifier_status_query"
    self.get_ai_mgr_status_service = rospy.ServiceProxy(AI_MGR_STATUS_SERVICE_NAME, ImageClassifierStatusQuery)
    time.sleep(1)
    rospy.Timer(rospy.Duration(1), self.updaterCb)

    # Start AI Manager Subscribers
    FOUND_OBJECT_TOPIC = NEPI_BASE_NAMESPACE + "ai_detector_mgr/found_object"
    rospy.Subscriber(FOUND_OBJECT_TOPIC, ObjectCount, self.found_object_callback, queue_size = 1)
    BOUNDING_BOXES_TOPIC = NEPI_BASE_NAMESPACE + "ai_detector_mgr/bounding_boxes"
    rospy.Subscriber(BOUNDING_BOXES_TOPIC, BoundingBoxes, self.object_detected_callback, queue_size = 1)
    DETECTION_IMAGE_TOPIC = NEPI_BASE_NAMESPACE + "ai_detector_mgr/detection_image"
    rospy.Subscriber(DETECTION_IMAGE_TOPIC, Image, self.detectionImageCb, queue_size = 1)

    #self.detection_targeting_image_pub = rospy.Publisher("~detection_image",Image,queue_size=1)
    # Setup Node Publishers
    self.status_pub = rospy.Publisher("~status", AiTargetingStatus, queue_size=1, latch=True)
    self.targeting_boxes_2d_pub = rospy.Publisher("~targeting_boxes_2d", BoundingBoxes, queue_size=1)
    self.targeting_boxes_3d_pub = rospy.Publisher("~targeting_boxes_3d", BoundingBoxes3D, queue_size=1)
    self.target_localizations_pub = rospy.Publisher("~targeting_localizations", TargetLocalizations, queue_size=1)
    self.alert_status_pub = rospy.Publisher("~alert_status",Bool,queue_size=1)
    self.alert_trigger_pub = rospy.Publisher("~alert_trigger",Bool,queue_size=1)
    #self.detection_image_pub = rospy.Publisher("~detection_image",Image,queue_size=1)
    self.targeting_image_pub = rospy.Publisher("~image",Image,queue_size=1)

    rospy.Timer(rospy.Duration(1), self.updateHasSubscribersThread)

    time.sleep(1)
    ## Initiation Complete
    rospy.loginfo("AI_TRG_APP: Initialization Complete")
    self.publish_status()

  def updateHasSubscribersThread(self,timer):
    #self.has_subscribers_detect_img = (self.detection_image_pub.get_num_connections() > 0)
    self.has_subscribers_target_img = (self.targeting_image_pub.get_num_connections() > 0)
  

  #######################
  ### AI Magnager Callbacks

  def updaterCb(self,timer):
    # Update status info from detector
    update_status = False
    # Purge Current Targets List based on Age
    current_timestamp = rospy.get_rostime()
    active_targets_dict = copy.deepcopy(self.active_targets_dict)
    lost_targets_dict = copy.deepcopy(self.lost_targets_dict)
    purge_list = []
    age_filter_sec = rospy.get_param('~target_age_filter', self.init_target_age_filter)
    #rospy.logwarn(active_targets_dict)
    for target in active_targets_dict.keys():
      last_timestamp = active_targets_dict[target]['last_detection_timestamp']
      #rospy.logwarn(target)
      #rospy.logwarn(ros_timestamp.to_sec())
      #rospy.logwarn(last_timestamp.to_sec())
      age =(current_timestamp.to_sec() - last_timestamp.to_sec())
      #rospy.logwarn("Target " + target + " age: " + str(age))
      if age > age_filter_sec:
        purge_list.append(target)
    #rospy.logwarn("Target Purge List: " + str(purge_list))
    for target in purge_list: 
        lost_targets_dict[target] = active_targets_dict[target]
        rospy.loginfo("AI_TRG_APP: Purging target: " + target + " from active target list")
        del active_targets_dict[target]
        update_status = True
    self.active_targets_dict = active_targets_dict
    self.lost_targets_dict = lost_targets_dict

    try:
      ai_mgr_status_response = self.get_ai_mgr_status_service()
    except Exception as e:
      rospy.loginfo("AI_TRG_APP: Failed to call AI MGR STATUS service " + str(e))
      return
    #status_str = str(ai_mgr_status_response)
    #rospy.logwarn("AI_TRG_APP: got ai manager status: " + status_str)
    self.current_image_topic = ai_mgr_status_response.selected_img_topic
    self.current_classifier = ai_mgr_status_response.selected_classifier
    self.current_classifier_state = ai_mgr_status_response.classifier_state
    classes_string = ai_mgr_status_response.selected_classifier_classes
    classes_list = nepi_ros.parse_string_list_msg_data(classes_string)
    if classes_list != self.classes_list:
      self.classes_list = classes_list
      self.current_classifier_classes = sorted(self.classes_list)
      if len(self.current_classifier_classes) > 0:
        cmap = plt.get_cmap('viridis')
        color_list = cmap(np.linspace(0, 1, len(self.current_classifier_classes))).tolist()
        rgb_list = []
        for color in color_list:
          rgb = []
          for i in range(3):
            rgb.append(int(color[i]*255))
          rgb_list.append(rgb)
        self.class_color_list = rgb_list
        #rospy.logwarn(self.class_color_list)
      #classes_str = str(self.current_classifier_classes)
      #rospy.logwarn("AI_TRG_APP: got ai manager status: " + classes_str)
      update_status = True
  
    selected_classes_dict = rospy.get_param('~selected_classes_dict', self.init_selected_classes_dict)
    last_classifier = rospy.get_param('~last_classiier', self.init_last_classifier)
    if last_classifier != self.current_classifier and self.current_classifier != "None":
      selected_classes_dict = dict() # Reset classes to all on new classifier
      for target_class in self.current_classifier_classes:
        selected_classes_dict[target_class] = {'depth': self.FACTORY_TARGET_DEPTH_METERS}
      update_status = True
    rospy.set_param('~selected_classes_dict', selected_classes_dict)
    rospy.set_param('~last_classiier', self.current_classifier)

    if (self.last_image_topic != self.current_image_topic) or (self.image_sub == None and self.current_image_topic != "None"):
      image_topic = nepi_ros.find_topic(self.current_image_topic)
      #rospy.logwarn(depth_map_topic)
      update_status = True
      if image_topic != "":
        self.current_image_topic = image_topic
        self.targeting_running = True
        update_status = True
        if self.image_sub != None:
          self.image_sub.unregister()
          self.image_sub = None
          time.sleep(1)
        rospy.loginfo("AI_TRG_APP: Subscribing to Image topic : " + image_topic)
        self.image_sub = rospy.Subscriber(image_topic, Image, self.targetingImageCb, queue_size = 1)
      
        # Look for Depth Map
        depth_map_topic = self.current_image_topic.rsplit('/',1)[0] + "/depth_map"
        depth_map_topic = nepi_ros.find_topic(depth_map_topic)
        if depth_map_topic == "":
          depth_map_topic = "None"
          self.has_depth_map = False
        else:
          self.has_depth_map = True
        self.depth_map_topic = depth_map_topic
        #rospy.logwarn(self.depth_map_topic)
        if depth_map_topic != "None":
          if self.depth_map_sub != None:
            self.depth_map_sub.unregister()
            self.depth_map_sub = None
            time.sleep(1)
          rospy.loginfo("AI_TRG_APP: Subscribing to Depth Map topic : " + depth_map_topic)
          self.depth_map_sub = rospy.Subscriber(depth_map_topic, Image, self.depthMapCb, queue_size = 10)
          update_status = True
          # If there is a depth_map, check for pointdcloud
          pointcloud_topic = self.current_image_topic.rsplit('/',1)[0] + "/pointcloud"
          pointcloud_topic = nepi_ros.find_topic(pointcloud_topic)
          if pointcloud_topic == "":
            pointcloud_topic = "None"
            self.has_pointcloud = False
          else:
            self.has_pointcloud = True
          self.pointcloud_topic = pointcloud_topic
                  

    elif self.current_image_topic == "None":  # Turn off targeting subscribers and reset last image topic
      self.targeting_running = False
      self.current_targets_dict = dict()
      if self.depth_map_sub != None:
        self.depth_map_sub.unregister()
        self.has_depth_map = False
        self.depth_map_header = Header()
      self.depth_map_topic = "None"
      self.has_pointcloud = False
      self.pointcloud_topic = "None"
      update_status = True
      time.sleep(1)

    self.last_image_topic = self.current_image_topic
    if update_status == True:
      self.publish_status()



  ### Monitor Output of AI model to clear detection status
  def found_object_callback(self,found_obj_msg):
    # Must reset target lists if no targets are detected
    if found_obj_msg.count == 0:
      #print("No objects detected")
      self.detect_boxes_msg = None
      self.targeting_box_3d_list = None
      self.current_targets_dict = dict()
      self.alert_status = False



  ### If object(s) detected, save bounding box info to global
  def object_detected_callback(self,bounding_boxes_msg):
    detect_header = bounding_boxes_msg.header
    ros_timestamp = bounding_boxes_msg.header.stamp
    image_seq_num = bounding_boxes_msg.header.seq
    self.detect_boxes_msg=bounding_boxes_msg
    self.save_data2file('ai_detector_mgr','detection_boxes',bounding_boxes_msg,ros_timestamp)
    transform = rospy.get_param('~idx/frame_3d_transform', self.init_frame_3d_transform)
    selected_classes_dict = rospy.get_param('~selected_classes_dict', self.init_selected_classes_dict)
    alert_classes = list(selected_classes_dict.keys())
    # Check for alert class
    alert = False
    for box in bounding_boxes_msg.bounding_boxes:
      if box.Class in alert_classes:
        alert = True
    if alert == True:
      self.alert_trigger_pub.publish(Empty)
    self.alert_status = alert    
    self.alert_status_pub.publish(alert)

    # Process targets
    active_targets_dict = copy.deepcopy(self.active_targets_dict)
    current_targets_dict = dict()
    bbs2d = []
    tls = []
    bbs3d = []
    if self.img_height != 0 and self.img_width != 0:
        # Iterate over all of the objects and calculate range and bearing data
        
        image_fov_vert = rospy.get_param('~image_fov_vert',  self.init_image_fov_vert)
        image_fov_horz = rospy.get_param('~image_fov_horz', self.init_image_fov_horz)
        target_box_adjust_percent = rospy.get_param('~target_box_percent',  self.init_target_box_adjust)
        target_min_points = rospy.get_param('~target_min_points',  self.init_target_min_points)    
        target_min_px_ratio = rospy.get_param('~target_min_px_ratio', self.init_target_min_px_ratio)
        target_min_dist_m = rospy.get_param('~target_min_dist_m', self.init_target_min_dist_m)
        detect_boxes_msg = copy.deepcopy(self.detect_boxes_msg)
        targeting_boxes_msg = copy.deepcopy(detect_boxes_msg)
        target_uids = []
        if detect_boxes_msg is not None:
            targeting_boxes_msg.bounding_boxes = [] 
            box_class_list = []
            box_area_list = []
            box_mmx_list = []
            box_mmy_list = []
            box_center_list = []
            for i, box in enumerate(detect_boxes_msg.bounding_boxes):
                box_class_list.append(box.Class)
                box_area=(box.xmax-box.xmin)*(box.ymax-box.ymin)
                box_area_list.append(box_area)
                box_mmx_list.append([box.xmin,box.xmax])
                box_mmy_list.append([box.ymin,box.ymax])
                box_y = box.ymin + (box.ymax - box.ymin)
                box_x = box.xmin + (box.xmax - box.xmin)
                box_center = [box_y,box_x]
                box_center_list.append(box_center)
            for i, box in enumerate(detect_boxes_msg.bounding_boxes):
                if box.Class in selected_classes_dict.keys():
                    self.seq += 1 
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
                    #rospy.logwarn([ymin_adj,ymax_adj,xmin_adj,xmax_adj])
                    # Calculate target range
                    target_range_m=float(-999)  # NEPI standard unset value
                    target_depth = selected_classes_dict[box.Class]['depth']
                    np_depth_array_m = copy.deepcopy(self.np_depth_array_m)
                    if np_depth_array_m is not None and self.depth_map_topic != "None":
                        
                        depth_map_header = copy.deepcopy(self.depth_map_header)
                        # Get target range from cropped and filtered depth data
                        depth_box_adj= np_depth_array_m[ymin_adj:ymax_adj,xmin_adj:xmax_adj]
                        depth_array=depth_box_adj.flatten()
                        depth_array = depth_array[~np.isnan(depth_array)] # remove nan entries
                        depth_array = depth_array[depth_array>0] # remove zero entries
                        depth_val=np.mean(depth_array) # Initialize fallback value.  maybe updated
                        #rospy.logwarn("got depth data")
                        #rospy.logwarn(depth_val)
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
                            #rospy.logwarn(max_hist_ind)
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
                                
                                #rospy.logwarn([min_range,max_range])
                                #rospy.logwarn(bins)
                                #rospy.logwarn(han_win)
                                if np.sum(han_win) > .1:
                                    depth_val=np.average(bins,weights = han_win)
                                


                        min_filter=depth_val-target_depth_m/2
                        max_filter=depth_val+target_depth_m/2
                        depth_array=depth_array[depth_array > min_filter]
                        depth_array=depth_array[depth_array < max_filter]
                        depth_len=len(depth_array)
                        #rospy.logwarn("")
                        #rospy.logwarn(depth_len)
                        if depth_len > target_min_points:
                            target_range_m=depth_val
                        else:
                            target_range_m= -999
                        #rospy.logwarn(target_range_m)
                        
                    # Calculate target bearings
                    object_loc_y_pix = float(box.ymin + ((box.ymax - box.ymin))  / 2) 
                    object_loc_x_pix = float(box.xmin + ((box.xmax - box.xmin))  / 2)
                    object_loc_y_ratio_from_center = float(object_loc_y_pix - self.img_height/2) / float(self.img_height/2)
                    object_loc_x_ratio_from_center = float(object_loc_x_pix - self.img_width/2) / float(self.img_width/2)
                    target_vert_angle_deg = (object_loc_y_ratio_from_center * float(image_fov_vert/2))
                    target_horz_angle_deg = - (object_loc_x_ratio_from_center * float(image_fov_horz/2))
                    ### Print the range and bearings for each detected object
                ##      print(target_label)
                ##      print(str(depth_box_adj.shape) + " detection box size")
                ##      print(str(depth_len) + " valid depth readings")
                ##      print("%.2f" % target_range_m + "m : " + "%.2f" % target_horz_angle_deg + "d : " + "%.2f" % target_vert_angle_deg + "d : ")
                ##      print("")

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
                    if valid_2d_target:
                        #### NEED TO Calculate Unique IDs
                        uid_suffix = 0
                        target_uid = box.Class + "_" + str(uid_suffix)# Need to add unque id tracking
                        while target_uid in target_uids:
                            uid_suffix += 1
                            target_uid = box.Class + "_" + str(uid_suffix)
                        target_uids.append(target_uid)
                        bounding_box_3d_msg = None
                        if self.selected_target == "All" or self.selected_target == target_uid:
                            # Updated Bounding Box 2d
                            bounding_box_msg = BoundingBox()
                            bounding_box_msg.Class = box.Class
                            bounding_box_msg.id = box.id
                            bounding_box_msg.uid = target_uid
                            bounding_box_msg.probability = box.probability
                            bbs2d.append(bounding_box_msg)


                            # Create target_localizations
                            target_data_msg=TargetLocalization()
                            target_data_msg.Class = box.Class
                            target_data_msg.id = box.id 
                            target_data_msg.uid = target_uid
                            target_data_msg.range_m=target_range_m
                            target_data_msg.azimuth_deg=target_horz_angle_deg
                            target_data_msg.elevation_deg=target_vert_angle_deg
                            tls.append(target_data_msg)

                            # Create Bounding Box 3d
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
                                bbc.y = target_range_m * math.sin(theta_rad) * math.sin(phi_rad) - transform[0]
                                bbc.z = target_range_m * math.cos(theta_rad) - transform[0]
                                #rospy.logwarn([target_range_m,theta_deg,phi_deg,bbc.x, bbc.y,bbc.z])
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
                                # Target Rotation Unknown
                                bbr = Vector3()
                                bbr.x = 0
                                bbr.y = 0
                                bbr.z = 0
                                bounding_box_3d_msg.box_rotation_rpy_deg.x = bbr.x
                                bounding_box_3d_msg.box_rotation_rpy_deg.y = bbr.y
                                bounding_box_3d_msg.box_rotation_rpy_deg.z = bbr.z
                                # To Do Add Bounding Box 3D Data
                                bbs3d.append(bounding_box_3d_msg)

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
                                'velocity_mps': [0,0,0],
                                'last_detection_timestamp': ros_timestamp                              
                                }
                            active_targets_dict[target_uid] = current_targets_dict[target_uid]

    self.active_targets_dict = active_targets_dict
    if current_targets_dict.keys() != self.current_targets_dict.keys():
        self.publish_status()
    self.current_targets_dict = current_targets_dict
    #rospy.logwarn(self.current_targets_dict)

    # Publish and Save 2D Bounding Boxes
    if len(bbs2d) > 0:
      detect_boxes_msg.bounding_boxes = bbs2d
      if not rospy.is_shutdown():
        self.targeting_boxes_2d_pub.publish(detect_boxes_msg)
      # Save Data if it is time.
      self.save_data2file('ai_targeting_app',"targeting_boxes_2d",detect_boxes_msg,ros_timestamp)

    # Publish and Save Target Localizations
    if len(tls) > 0:
      target_locs_msg = TargetLocalizations()
      target_locs_msg.header = detect_header
      target_locs_msg.image_topic = self.current_image_topic
      target_locs_msg.image_header = self.current_image_header
      target_locs_msg.depth_topic = self.depth_map_topic
      target_locs_msg.depth_header = self.depth_map_header
      target_locs_msg.target_localizations = tls
      if not rospy.is_shutdown():
        self.target_localizations_pub.publish(target_locs_msg)
      # Save Data if Time
      self.save_data2file('ai_targeting_app','targeting_localizations',target_locs_msg,ros_timestamp)

    # Publish and Save 3D Bounding Boxes
    self.targeting_box_3d_list = bbs3d
    #rospy.logwarn("")
    #rospy.logwarn(bbs3d)
    if len(bbs3d) > 0:
      targeting_boxes_3d_msg = BoundingBoxes3D()
      targeting_boxes_3d_msg.header = detect_header
      targeting_boxes_3d_msg.image_topic = self.current_image_topic
      targeting_boxes_3d_msg.image_header = self.current_image_header
      targeting_boxes_3d_msg.depth_map_header = self.depth_map_header
      targeting_boxes_3d_msg.depth_map_topic = self.depth_map_topic
      targeting_boxes_3d_msg.bounding_boxes_3d = bbs3d
      if not rospy.is_shutdown():
        self.targeting_boxes_3d_pub.publish(targeting_boxes_3d_msg)
      # Save Data if Time
      self.save_data2file('ai_targeting_app','targeting_boxes_3d',targeting_boxes_3d_msg,ros_timestamp)

  def detectionImageCb(self,img_in_msg):
    data_product = 'image'
    output_image = rospy.get_param('~selected_output_image', self.init_selected_output_image)
    if output_image == 'detection_image':
      has_subscribers =  self.has_subscribers_target_img
      saving_is_enabled = self.save_data_if.data_product_saving_enabled(data_product)
      data_should_save  = self.save_data_if.data_product_should_save(data_product) and saving_is_enabled
      snapshot_enabled = self.save_data_if.data_product_snapshot_enabled(data_product)
      save_data = data_should_save or snapshot_enabled
      ros_timestamp = img_in_msg.header.stamp
      # Publish new image to ros
      if not rospy.is_shutdown() and has_subscribers: #and has_subscribers:
          #Convert OpenCV image to ROS image
          self.targeting_image_pub.publish(img_in_msg)
      # Save Data if Time
      if save_data:
          cv2_img = nepi_img.rosimg_to_cv2img(img_in_msg)
          self.save_img2file('ai_targeting_app',data_product,cv2_img,ros_timestamp)


  def targetingImageCb(self,img_in_msg):    
    data_product = 'image'
    output_image = rospy.get_param('~selected_output_image', self.init_selected_output_image)
    if output_image == 'alert_image' or output_image == 'targeting_image':
      has_subscribers =  self.has_subscribers_target_img
      saving_is_enabled = self.save_data_if.data_product_saving_enabled(data_product)
      data_should_save  = self.save_data_if.data_product_should_save(data_product) and saving_is_enabled
      snapshot_enabled = self.save_data_if.data_product_snapshot_enabled(data_product)
      save_data = data_should_save or snapshot_enabled

      if save_data or has_subscribers:
        self.current_image_header = img_in_msg.header
        ros_timestamp = img_in_msg.header.stamp     
        self.img_height = img_in_msg.height
        self.img_width = img_in_msg.width
        cv2_img = nepi_img.rosimg_to_cv2img(img_in_msg)
        target_dict = copy.deepcopy(self.current_targets_dict)

        # Process Targeting Image if Needed
        if target_dict == None:
            target_dict = dict()
        if len(target_dict.keys()) > 0:
            for target_uid in target_dict.keys():
                #rospy.logwarn(target_dict[target_uid])
                target = target_dict[target_uid]
                class_name = target['class_name']
                [target_range_m , target_horz_angle_deg , target_vert_angle_deg] = target['range_bearings']
                ###### Apply Image Overlays and Publish Targeting_Image ROS Message
                # Overlay adjusted detection boxes on image 
                [xmin,xmax,ymin,ymax] = target_dict[target_uid]["bounding_box"]
                start_point = (xmin, ymin)
                end_point = (xmax, ymax)
                class_name = class_name
                
                if output_image == 'alert_image':
                  class_color = (0,0,255)
                  line_thickness = 2
                  cv2.rectangle(cv2_img, start_point, end_point, color=class_color, thickness=line_thickness)
                else:
                  class_color = (255,0,0)
                  if class_name in self.current_classifier_classes:
                      class_ind = self.current_classifier_classes.index(class_name)
                      if class_ind < len(self.class_color_list):
                          class_color = tuple(self.class_color_list[class_ind])
                  line_thickness = 2
                  cv2.rectangle(cv2_img, start_point, end_point, color=class_color, thickness=line_thickness)
                  # Overlay text data on OpenCV image
                  font                   = cv2.FONT_HERSHEY_DUPLEX
                  fontScale, thickness  = self.optimal_font_dims(cv2_img,font_scale = 1.2e-3, thickness_scale = 1.5e-3)
                  fontColor = (0, 255, 0)
                  lineType = 1
                  text_size = cv2.getTextSize("Text", 
                      font, 
                      fontScale,
                      thickness)
                  line_height = text_size[1] * 3
                  # Overlay Label
                  text2overlay=target_uid
                  bottomLeftCornerOfText = (xmin + line_thickness,ymin + line_thickness * 2 + line_height)
                  cv2.putText(cv2_img,text2overlay, 
                      bottomLeftCornerOfText, 
                      font, 
                      fontScale,
                      fontColor,
                      thickness,
                      lineType)
                
                  # Overlay Data
                  #rospy.logwarn(line_height)
                  if target_range_m == -999:
                      text2overlay="#," + "%.f" % target_horz_angle_deg + "d," + "%.f" % target_vert_angle_deg + "d"
                  else:
                      text2overlay="%.1f" % target_range_m + "m," + "%.f" % target_horz_angle_deg + "d," + "%.f" % target_vert_angle_deg + "d"
                  bottomLeftCornerOfText = (xmin + line_thickness,ymin + line_thickness * 2 + line_height * 2)
                  cv2.putText(cv2_img,text2overlay, 
                      bottomLeftCornerOfText, 
                      font, 
                      fontScale,
                      fontColor,
                      thickness,
                      lineType)   
        # Publish new image to ros
        if not rospy.is_shutdown() and has_subscribers: #and has_subscribers:
            #Convert OpenCV image to ROS image
            img_out_msg = nepi_img.cv2img_to_rosimg(cv2_img, encoding='bgr8')
            self.targeting_image_pub.publish(img_out_msg)
        # Save Data if Time
        if save_data:
            self.save_img2file('ai_targeting_app',data_product,cv2_img,ros_timestamp)


  def optimal_font_dims(self, img, font_scale = 2e-3, thickness_scale = 5e-3):
    h, w, _ = img.shape
    font_scale = min(w, h) * font_scale
    thickness = math.ceil(min(w, h) * thickness_scale)
    return font_scale, thickness

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

  ###################
  ## Status Publisher
  def publish_status(self):
    status_msg = AiTargetingStatus()

    status_msg.targeting_running = self.targeting_running

    status_msg.classifier_name = self.current_classifier
    status_msg.classifier_state = self.current_classifier_state

    status_msg.image_topic = self.current_image_topic
    status_msg.has_depth_map = self.has_depth_map
    status_msg.depth_map_topic = self.depth_map_topic
    status_msg.has_pointcloud = self.has_pointcloud
    status_msg.pointcloud_topic = self.pointcloud_topic

    status_msg.output_image_options_list = str(self.output_image_options)
    status_msg.selected_output_image = rospy.get_param('~selected_output_image', self.init_selected_output_image)

    status_msg.available_classes_list = str(self.current_classifier_classes)
    selected_classes_dict = rospy.get_param('~selected_classes_dict', self.init_selected_classes_dict)
    classes_list = []
    depth_list = []
    for key in selected_classes_dict.keys():
      classes_list.append(key)
      depth_list.append(str(selected_classes_dict[key]['depth']))
    status_msg.selected_classes_list = str(classes_list)
    status_msg.selected_classes_depth_list = str(depth_list)

    status_msg.alert_status = self.alert_status

    targets_list = self.active_targets_dict.keys()
    avail_targets_list = []
    if self.selected_target != "All" and self.selected_target not in targets_list:
      avail_targets_list.append(self.selected_target)
    for target in targets_list:
      avail_targets_list.append(target) 
    avail_targets_list = sorted(avail_targets_list)
    avail_targets_list.insert(0,"All")
    status_msg.available_targets_list = str(avail_targets_list)
    status_msg.selected_target = self.selected_target


    status_msg.image_fov_vert_degs = rospy.get_param('~image_fov_vert',  self.init_image_fov_vert)
    status_msg.image_fov_horz_degs = rospy.get_param('~image_fov_horz', self.init_image_fov_horz)

    status_msg.target_box_size_percent = rospy.get_param('~target_box_percent',  self.init_target_box_adjust)
    status_msg.default_target_depth_m = rospy.get_param('~default_target_depth',  self.init_default_target_depth)
    status_msg.target_min_points = rospy.get_param('~target_min_points', self.init_target_min_points)
    status_msg.target_min_px_ratio = rospy.get_param('~target_min_px_ratio', self.init_target_min_px_ratio)
    status_msg.target_min_dist_m = rospy.get_param('~target_min_dist_m', self.init_target_min_dist_m)
    status_msg.target_age_filter = rospy.get_param('~target_age_filter', self.init_target_age_filter)
    # The transfer frame into which 3D data (pointclouds) are transformed for the pointcloud data topic
    transform_3d = rospy.get_param('~frame_3d_transform',  self.ZERO_TRANSFORM)
    status_msg.transform_3d = str(transform_3d)

    self.status_pub.publish(status_msg)

 
      
    
  #######################
  # Data Saving Funcitons
 



  def save_data2file(self,node_name,data_product,data,ros_timestamp):
      if self.save_data_if is not None:
        saving_is_enabled = self.save_data_if.data_product_saving_enabled(data_product)
        snapshot_enabled = self.save_data_if.data_product_snapshot_enabled(data_product)
        if data is not None:
            if (self.save_data_if.data_product_should_save(data_product) or snapshot_enabled):
                full_path_filename = self.save_data_if.get_full_path_filename(nepi_ros.get_datetime_str_from_stamp(ros_timestamp), 
                                                                                        node_name + "_" + data_product, 'txt')
                if os.path.isfile(full_path_filename) is False:
                    with open(full_path_filename, 'w') as f:
                        f.write('input_image: ' + self.current_image_topic + '\n')
                        f.write(str(data))
                        self.save_data_if.data_product_snapshot_reset(data_product)

  def save_img2file(self,node_name,data_product,cv2_img,ros_timestamp):
      if self.save_data_if is not None:
        if cv2_img is not None:
            full_path_filename = self.save_data_if.get_full_path_filename(nepi_ros.get_datetime_str_from_stamp(ros_timestamp), 
                                                                                    node_name + "_" + data_product, 'png')
            if os.path.isfile(full_path_filename) is False:
                cv2.imwrite(full_path_filename, cv2_img)
                self.save_data_if.data_product_snapshot_reset(data_product)

  def save_pc2file(self,node_name,data_product,o3d_pc,ros_timestamp):
      if self.save_data_if is not None:
        if o3d_pc is not None:
            full_path_filename = self.save_data_if.get_full_path_filename(nepi_ros.get_datetime_str_from_stamp(ros_timestamp), 
                                                                                    node_name + "_" + data_product, 'pcd')
            if os.path.isfile(full_path_filename) is False:
                nepi_pc.save_pointcloud(o3d_pc,full_path_filename)
                self.save_data_if.data_product_snapshot_reset(data_product)

                
    
  #######################
  # Node Cleanup Function
  
  def cleanup_actions(self):
    rospy.loginfo("AI_TRG_APP: Shutting down: Executing script cleanup actions")


#########################################
# Main
#########################################
if __name__ == '__main__':
  node_name = "ai_targeting_app"
  rospy.init_node(name=node_name)
  #Launch the node
  rospy.loginfo("AI_TRG_APP: Launching node named: " + node_name)
  node = NepiAiTargetingApp()
  #Set up node shutdown
  rospy.on_shutdown(node.cleanup_actions)
  # Spin forever (until object is detected)
  rospy.spin()






