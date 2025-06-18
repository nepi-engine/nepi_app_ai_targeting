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
from nepi_interfaces.msg import BoundingBox, BoundingBoxes, BoundingBox3D, BoundingBoxes3D, ObjectCount
from nepi_interfaces.msg import StringArray, TargetLocalization, TargetLocalizations
from nepi_interfaces.msg import Frame3DTransform
from nepi_app_ai_targeting.msg import AiTargetingStatus, AiTargetingTargets


from nepi_api.node_if import NodeClassIF
from nepi_api.messages_if import MsgIF
from nepi_api.connect_node_if import ConnectNodeClassIF
from nepi_api.system_if import SaveDataIF
from nepi_api.system_if import SaveCfgIF
from nepi_api.data_if import ImageIF

from nepi_api.connect_node_if import ConnectNodeClassIF

APP_NODE_NAME = 'app_ai_targeting'

class ConnectAppAITargeting:
    msg_if = None
    ready = False
    namespace = '~'

    con_node_if = None

    connected = False
    status_msg = None
    status_connected = False

 
    #######################
    ### IF Initialization
    def __init__(self, 
                namespace = None,
                ):
        ####  IF INIT SETUP ####
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

        if namespace is None:
            namespace = nepi_sdk.create_namespace(self.base_namespace,APP_NODE_NAME)
        else:
            namespace = namespace
        self.namespace = nepi_sdk.get_full_namespace(namespace)


        ##############################   
        ## Node Setup

        # Configs Config Dict ####################
        self.CFGS_DICT = {
                'namespace': self.namespace
        }


        # Services Config Dict ####################
        self.SRVS_DICT = None


        # Publishers Config Dict ####################
        self.PUBS_DICT = {
            'publish_status': {
                'namespace': self.node_namespace,
                'topic': 'publish_status',
                'msg': Empty,
                'qsize': 10,
                'latch': False
            },
            'enable_app': {
                'namespace': self.node_namespace,
                'topic': 'enable_app',
                'msg': Bool,
                'qsize': 10,
                'latch': False
            },
            'set_image_fov_vert': {
                'namespace': self.node_namespace,
                'topic': 'set_image_fov_vert',
                'msg': Float32,
                'qsize': 10,
                'latch': False
            },
            'set_image_fov_horz': {
                'namespace': self.node_namespace,
                'topic': 'set_image_fov_horz',
                'msg': Float32,
                'qsize': 10,
                'latch': False
            },
            'add_all_target_classes': {
                'namespace': self.node_namespace,
                'topic': 'add_all_target_classes',
                'msg': Empty,
                'qsize': 10,
                'latch': False
            },
            'add_target_class': {
                'namespace': self.node_namespace,
                'topic': 'add_target_class',
                'msg': String,
                'qsize': 10,
                'latch': False
            },
            'remove_target_class': {
                'namespace': self.node_namespace,
                'topic': 'remove_target_class',
                'msg': String,
                'qsize': 10,
                'latch': False
            },
            'select_target': {
                'namespace': self.node_namespace,
                'topic': 'select_target',
                'msg': String,
                'qsize': 10,
                'latch': False
            },
            'set_target_box_size_percent': {
                'namespace': self.node_namespace,
                'topic': 'set_target_box_size_percent',
                'msg': Int32,
                'qsize': 10,
                'latch': False
            },
            'set_default_target_detpth': {
                'namespace': self.node_namespace,
                'topic': 'set_default_target_detpth',
                'msg': Float32,
                'qsize': 10,
                'latch': False
            },
            'set_target_min_points': {
                'namespace': self.node_namespace,
                'topic': 'set_target_min_points',
                'msg': Int32,
                'qsize': 10,
                'latch': False
            },
            'set_target_min_px_ratio': {
                'namespace': self.node_namespace,
                'topic': 'set_target_min_px_ratio',
                'msg': Float32,
                'qsize': 10,
                'latch': False
            },
            'set_age_filter': {
                'namespace': self.node_namespace,
                'topic': 'set_age_filter',
                'msg': Float32,
                'qsize': 10,
                'latch': False
            },
            'set_frame_3d_transform': {
                'namespace': self.node_namespace,
                'topic': 'set_frame_3d_transform',
                'msg': Frame3DTransform,
                'qsize': 10,
                'latch': False
            },
            'clear_frame_3d_transform': {
                'namespace': self.node_namespace,
                'topic': 'clear_frame_3d_transform',
                'msg': Empty,
                'qsize': 10,
                'latch': False
            },
            'found_object': {
                'namespace': self.node_namespace,
                'topic': '/found_object', #self.ai_mgr_namespace  + "/found_object"
                'msg': ObjectCount,
                'qsize': 1,
                'latch': False
            },
            'bounding_boxes': {
                'namespace': self.node_namespace,
                'topic': '/bounding_boxes', #self.ai_mgr_namespace  + "/bounding_boxes"
                'msg': BoundingBoxes,
                'qsize': 1,
                'latch': False
            },
            'image_topic': {
                'namespace': self.node_namespace,
                'topic': '/image_topic',
                'msg': Image,
                'qsize': 1,
                'latch': False
            },
            'depth_map_topic': {
                'namespace': self.node_namespace,
                'topic': '/depth_map_topic',
                'msg': Image,
                'qsize': 10,
                'latch': False
            }
        }


        # Subscribers Config Dict ####################
        self.SUBS_DICT = {
            'status': {
                'namespace': self.node_namespace,
                'topic': 'status',
                'msg': AiTargetingStatus,
                'qsize': 1,
                'callback': self._statusCb
            },
            'targets': {
                'namespace': self.node_namespace,
                'topic': 'targets',
                'msg': AiTargetingTargets,
                'qsize': 1,
                'callback': self._statusCb
            },
            'boxes_count': {
                'namespace': self.node_namespace,
                'topic': 'boxes_count',
                'msg': ObjectCount,
                'qsize': 1,
                'callback': self._statusCb
            },
            'boxes3d_count': {
                'namespace': self.node_namespace,
                'topic': 'boxes3d_count',
                'msg': ObjectCount,
                'qsize': 1,
                'callback': self._statusCb
            },
            'target_count': {
                'namespace': self.node_namespace,
                'topic': 'target_count',
                'msg': ObjectCount,
                'qsize': 1,
                'callback': self._statusCb
            },
            'target_boxes_2d': {
                'namespace': self.node_namespace,
                'topic': 'target_boxes_2d',
                'msg': BoundingBoxes,
                'qsize': 1,
                'callback': self._statusCb
            },
            'target_boxes_3d': {
                'namespace': self.node_namespace,
                'topic': 'target_boxes_3d',
                'msg': BoundingBoxes3D,
                'qsize': 1,
                'callback': self._statusCb
            },
            'target_localizations': {
                'namespace': self.node_namespace,
                'topic': 'target_localizations',
                'msg': TargetLocalizations,
                'qsize': 1,
                'callback': self._statusCb
            }



        }


        # Create Node Class ####################
        
        self.con_node_if = NodeClassIF(
                        configs_dict = self.CFGS_DICT,
                        services_dict = self.SRVS_DICT,
                        pubs_dict = self.PUBS_DICT,
                        subs_dict = self.SUBS_DICT,
                        log_class_name = True,
                        msg_if = self.msg_if
        )

        

        self.con_node_if.wait_for_ready()


        ##############################
        # Complete Initialization
        self.ready = True
        self.msg_if.pub_info("IF Initialization Complete")
        ###############################
    

    #######################
    # Class Public Methods
    #######################


    def get_ready_state(self):
        return self.ready

    def wait_for_ready(self, timout = float('inf') ):
        success = False
        if self.ready is not None:
            self.msg_if.pub_info("Waiting for connection")
            timer = 0
            time_start = nepi_sdk.get_time()
            while self.ready == False and timer < timeout and not nepi_sdk.is_shutdown():
                nepi_sdk.sleep(.1)
                timer = nepi_sdk.get_time() - time_start
            if self.ready == False:
                self.msg_if.pub_info("Failed to Connect")
            else:
                self.msg_if.pub_info("Connected")
        return self.ready  

    def get_namespace(self):
        return self.namespace

    def check_connection(self):
        return self.connected

    def wait_for_connection(self, timout = float('inf') ):
        if self.con_node_if is not None:
            self.msg_if.pub_info("Waiting for connection")
            timer = 0
            time_start = nepi_sdk.get_time()
            while self.connected == False and timer < timeout and not nepi_sdk.is_shutdown():
                nepi_sdk.sleep(.1)
                timer = nepi_sdk.get_time() - time_start
            if self.connected == False:
                self.msg_if.pub_info("Failed to Connect")
            else:
                self.msg_if.pub_info("Connected")
        return self.connected


    def check_status_connection(self):
        return self.status_connected

    def wait_for_status_connection(self, timout = float('inf') ):
        if self.con_node_if is not None:
            self.msg_if.pub_info("Waiting for status connection")
            timer = 0
            time_start = nepi_sdk.get_time()
            while self.status_connected == False and timer < timeout and not nepi_sdk.is_shutdown():
                nepi_sdk.sleep(.1)
                timer = nepi_sdk.get_time() - time_start
            if self.status_connected == False:
                self.msg_if.pub_info("Failed to connect to status msg")
            else:
                self.msg_if.pub_info("Status Connected")
        return self.status_connected

    def get_status_dict(self):
        img_status_dict = None
        if self.status_msg is not None:
            img_status_dict = nepi_sdk.convert_msg2dict(self.status_msg)
        return self.img_status_dict

    def unregister(self):
        self._unsubscribeTopic()

    def publish_status(self):
        pub_name = 'publish_status'
        msg = Empty()
        self.con_node_if.publish_pub(pub_name,msg)

    def enable_app(self,enable_app):
        pub_name = 'enable_app'
        msg = enable_app
        self.con_node_if.publish_pub(pub_name,msg)  

    def set_image_fov_vert(self,fov_vert):
        pub_name = 'set_image_fov_vert'
        msg = fov_vert
        self.con_node_if.publish_pub(pub_name,msg) 

    def set_image_fov_horz(self,fov_vert):
        pub_name = 'set_image_fov_horz'
        msg = fov_vert
        self.con_node_if.publish_pub(pub_name,msg) 

    def enable_publishing(self):
        pub_name = 'start_pub'
        msg = Empty()
        self.con_node_if.publish_pub(pub_name,msg)

    def add_all_target_classes(self,classes):
        pub_name = 'add_all_target_classes'
        msg = classes
        self.con_node_if.publish_pub(pub_name,msg) 

    def add_target_class(self,target_class):
        pub_name = 'add_target_class'
        msg = target_class
        self.con_node_if.publish_pub(pub_name,msg) 

    def remove_target_class(self,target_class):
        pub_name = 'remove_target_class'
        msg = target_class
        self.con_node_if.publish_pub(pub_name,msg) 

    def select_target(self,target):
        pub_name = 'select_target'
        msg = target
        self.con_node_if.publish_pub(pub_name,msg) 

    def set_target_box_size_percent(self,size_percent):
        pub_name = 'set_target_box_size_percent'
        msg = size_percent
        self.con_node_if.publish_pub(pub_name,msg) 

    def set_default_target_detpth(self,target_detpth):
        pub_name = 'set_default_target_detpth'
        msg = target_detpth
        self.con_node_if.publish_pub(pub_name,msg) 

    def set_target_min_points(self,min_points):
        pub_name = 'set_target_min_points'
        msg = min_points
        self.con_node_if.publish_pub(pub_name,msg) 

    def set_target_min_px_ratio(self,px_ratio):
        pub_name = 'set_target_min_px_ratio'
        msg = px_ratio
        self.con_node_if.publish_pub(pub_name,msg) 

    def set_age_filter(self,age_filter):
        pub_name = 'set_age_filter'
        msg = age_filter
        self.con_node_if.publish_pub(pub_name,msg) 

    def set_frame_3d_transform(self,frame_3d):
        pub_name = 'set_frame_3d_transform'
        msg = frame_3d
        self.con_node_if.publish_pub(pub_name,msg) 

    def clear_frame_3d_transform(self):
        pub_name = 'clear_frame_3d_transform'
        msg = Empty()
        self.con_node_if.publish_pub(pub_name,msg)

    def found_object(self,found_object):
        pub_name = 'found_object'
        msg = found_object
        self.con_node_if.publish_pub(pub_name,msg) 

    def set_bounding_boxes(self,bounding_boxes):
        pub_name = 'bounding_boxes'
        msg = bounding_boxes
        self.con_node_if.publish_pub(pub_name,msg) 

    def image_topic(self,image_topic):
        pub_name = 'image_topic'
        msg = image_topic
        self.con_node_if.publish_pub(pub_name,msg) 

    def depth_map_topic(self,topic):
        pub_name = 'depth_map_topic'
        msg = topic
        self.con_node_if.publish_pub(pub_name,msg) 


    ###############################
    # Class Private Methods
    ###############################
   

    def _unsubscribeTopic(self):
        success = False
        self.connected = False
        if self.con_node_if is not None:
            self.msg_if.pub_warn("Unregistering topic: " + str(self.namespace))
            try:
                self.con_node_if.unregister_class()
                time.sleep(1)
                self.con_node_if = None
                self.namespace = None
                self.status_connected = False 
                self.data_dict = None
                success = True
            except Exception as e:
                self.msg_if.pub_warn("Failed to unregister image:  " + str(e))
        return success


    def _statusCb(self,status_msg):      
        self.status_connected = True
        self.status_msg = status_msg
