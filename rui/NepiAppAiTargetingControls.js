/*
 * Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.
 *
 * This file is part of nepi-engine
 * (see https://github.com/nepi-engine).
 *
 * License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
 */
import React, { Component } from "react"
import { observer, inject } from "mobx-react"

import Section from "./Section"
//import EnableAdjustment from "./EnableAdjustment"
import Button, { ButtonMenu } from "./Button"
import {SliderAdjustment} from "./AdjustmentWidgets"
import Label from "./Label"
import { Column, Columns } from "./Columns"
import Input from "./Input"
import Select, { Option } from "./Select"
import Styles from "./Styles"
import Toggle from "react-toggle"


import {round, convertStrToStrList, createMenuListFromStrList, onDropdownSelectedSendStr, onUpdateSetStateValue, onEnterSendFloatValue, onEnterSendIntValue, onEnterSetStateFloatValue} from "./Utilities"

@inject("ros")
@observer

// Component that contains the  Pointcloud App Viewer Controls
class NepiAppAiTargetingControls extends Component {
  constructor(props) {
    super(props)

    // these states track the values through  Status messages
    this.state = {
      classifier_name: null,
      classifier_state: null,

      image_topic: null,
      depth_map_topic: null,
      pointcloud_topic: null,

      available_classes_list: [],
      last_classes_list: [],
      selected_classes_list:[],
      selected_classes_depth_list: [],

      available_targets_list: [],
      selected_target: null,

      image_fov_vert_degs: null,
      image_fov_horz_degs: null,

      target_box_reduction_percent: null,
      default_target_depth_m: null,
      target_min_points: null,
      target_min_px_ratio: null,
      target_min_dist_m: null,
      target_age_filter: null,

      target_box_size_percent: null,
      pc_box_size_percent: null,

      current_targets_list: [],
      lost_targets_list: [],
      
      
      targetingListener: null,
      viewableTopics: false,

      output_image_options_list: [],
      selected_output_image: 'targeting_image',

      transforms_topic_list: [],
      transforms_list: [],
      transformTX: 0,
      transformTY: 0,
      transformTZ: 0,
      transformRX: 0,
      transformRY: 0,
      transformRZ: 0,
      transformHO: 0,
      showTransforms: false,
      update_transform: false,



      connected: false,
      needs_update: true

    }
  
    this.targetingListenerFunc = this.targetingListenerFunc.bind(this)
    this.updatetargetingListenerFunc = this.updatetargetingListenerFunc.bind(this)
    this.onToggleClassSelection = this.onToggleClassSelection.bind(this)
    this.getClassOptions = this.getClassOptions.bind(this)
    this.toggleViewableTopics = this.toggleViewableTopics.bind(this)
    this.sendTransformUpdateMessage = this.sendTransformUpdateMessage.bind(this)
    this.onClickToggleShowTransforms = this.onClickToggleShowTransforms.bind(this)
    this.sendClearTransformUpdateMessage = this.sendClearTransformUpdateMessage.bind(this)



  }

  // Callback for handling ROS Status messages
  targetingListenerFunc(message) {
    const avail_classes_str_list = convertStrToStrList(message.available_classes_list)
    this.setState({
    classifier_name: message.classifier_name,
    classifier_state: message.classifier_state,
    image_topic: message.image_topic,
    depth_map_topic: message.depth_map_topic,
    pointcloud_topic: message.pointcloud_topic,
    selected_target: message.selected_target,
    image_fov_vert_degs: message.image_fov_vert_degs,
    image_fov_horz_degs: message.image_fov_horz_degs,
    target_box_reduction_percent: message.target_box_reduction_percent,
    default_target_depth_m: message.default_target_depth_m,
    target_min_points: message.target_min_points,
    target_min_px_ratio: message.target_min_px_ratio,
    target_min_dist_m: message.target_min_dist_m,
    target_age_filter: message.target_age_filter,
    target_box_size_percent: message.target_box_size_percent,
    pc_box_size_percent: message.pointcloud_box_size_percent,

    output_image_options_list: convertStrToStrList(message.output_image_options_list),
    selected_output_image: message.selected_output_image,

    available_classes_list: avail_classes_str_list,
    selected_classes_list: convertStrToStrList(message.selected_classes_list),
    selected_classes_depth_list: convertStrToStrList(message.selected_classes_depth_list),
    available_targets_list: convertStrToStrList(message.available_targets_list),
    current_targets_list: convertStrToStrList(message.current_targets_list),
    lost_targets_list: convertStrToStrList(message.lost_targets_list),
        })

    if (this.state.update_transform === true){
        this.setState({
          transformTX: message.frame_3d_transform.translate_vector.x,
          transformTY: message.frame_3d_transform.translate_vector.y,
          transformTZ: message.frame_3d_transform.translate_vector.z,
          transformRX: message.frame_3d_transform.rotate_vector.x,
          transformRY: message.frame_3d_transform.rotate_vector.y,
          transformRZ: message.frame_3d_transform.rotate_vector.z,
          transformHO: message.frame_3d_transform.heading_offset,
      })
    }
    this.setState({
      update_transform: false
    })

    this.setState({
      connected: true
    })

    const last_classes_list = this.state.last_classes_list
    this.setState({
      last_classes_list: avail_classes_str_list
    })
    if (last_classes_list !== avail_classes_str_list){
      this.render()
    }
  }

    // Function for configuring and subscribing to Status
    updatetargetingListenerFunc() {
      const statusNamespace = this.props.targetingNamespace + '/status'
      if (this.state.targetingListener) {
        this.state.targetingListener.unsubscribe()
      }
      var targetingListener = this.props.ros.setupStatusListener(
            statusNamespace,
            "nepi_app_ai_targeting/AiTargetingStatus",
            this.targetingListenerFunc
          )
      this.setState({ 
        targetingListener: targetingListener,
        needs_update: false
      })
    }

  // Lifecycle method called when compnent updates.
  // Used to track changes in the topic
  componentDidUpdate(prevProps, prevState, snapshot) {
    const { targetingNamespace } = this.props
    const namespace_updated = (prevProps.targetingNamespace !== targetingNamespace && targetingNamespace !== null)
    const needs_update = (this.state.needs_update && targetingNamespace !== null)
    if (namespace_updated || needs_update) {
      if (targetingNamespace.indexOf('null') === -1){
        this.updatetargetingListenerFunc()
        this.render()
      } 
    }
  }


  // Lifecycle method called just before the component umounts.
  // Used to unsubscribe to Status message
  componentWillUnmount() {
    if (this.state.targetingListener) {
      this.state.targetingListener.unsubscribe()
    }
  }



  settransform(event){
    const pointcloud = event.target.value
    const pointclouds = this.state.transforms_topic_list
    const transforms = this.state.transforms_list
    const tf_index = pointclouds.indexOf(pointcloud)
    if (tf_index !== -1){
      this.setState({
        transformPointcloud: pointcloud,
        transformInd: tf_index
      })
      const transform = transforms[tf_index]
      this.setState({
        transformTX: round(transform[0]),
        transformTY: round(transform[1]),
        transformTZ: round(transform[2]),
        transformRX: round(transform[3]),
        transformRY: round(transform[4]),
        transformRZ: round(transform[5]),
        transformHO: round(transform[6])
      })
      
    }
  }

  sendTransformUpdateMessage(){
    const {sendFrame3DTransformMsg} = this.props.ros
    const namespace = this.props.targetingNamespace + "/set_frame_3d_transform"
    const TX = parseFloat(this.state.transformTX)
    const TY = parseFloat(this.state.transformTY)
    const TZ = parseFloat(this.state.transformTZ)
    const RX = parseFloat(this.state.transformRX)
    const RY = parseFloat(this.state.transformRY)
    const RZ = parseFloat(this.state.transformRZ)
    const HO = parseFloat(this.state.transformHO)
    const transformList = [TX,TY,TZ,RX,RY,RZ,HO]
    sendFrame3DTransformMsg(namespace,transformList)
    this.setState({
      update_transform: true
    })
  }

  sendClearTransformUpdateMessage(){
    const {sendClearFrame3DTransformMsg} = this.props.ros
    const namespace = this.props.idxSensorNamespace + "/set_frame_3d_transform"
    const transformList = [0,0,0,0,0,0,0]
    sendClearFrame3DTransformMsg(namespace,transformList)
    this.setState({
      update_transform: true
    })
  }






  // Function for creating image topic options.
  getClassOptions() {
  const classesList = this.state.available_classes_list
  var items = []
  items.push(<Option>{"None"}</Option>)
  items.push(<Option>{"All"}</Option>)
  if (classesList){
    for (var i = 0; i < classesList.length; i++) {
        items.push(<Option value={classesList[i]}>{classesList[i]}</Option>)
    }
  }
  return items
  }


  toggleViewableTopics() {
    const set = !this.state.viewableTopics
    this.setState({viewableTopics: set})
  }


  onToggleClassSelection(event){
    const {sendTriggerMsg, sendStringMsg} = this.props.ros
    const classSelection = event.target.value
    const selectedClassesList = this.state.selected_classes_list
    const addAllNamespace = this.props.targetingNamespace + "/add_all_target_classes"
    const removeAllNamespace = this.props.targetingNamespace + "/remove_all_target_classes"
    const addNamespace = this.props.targetingNamespace + "/add_target_class"
    const removeNamespace = this.props.targetingNamespace + "/remove_target_class"
    if (this.props.targetingNamespace){
      if (classSelection === "None"){
          sendTriggerMsg(removeAllNamespace)
      }
      else if (classSelection === "All"){
        sendTriggerMsg(addAllNamespace)
    }
      else if (selectedClassesList.indexOf(classSelection) !== -1){
        sendStringMsg(removeNamespace,classSelection)
      }
      else {
        sendStringMsg(addNamespace,classSelection)
      }
    }
  }

  onClickToggleShowTransforms(){
    const newVal = this.state.showTransforms === false
    this.setState({showTransforms: newVal})
    this.render()
  }

  render() {
    
    const {  sendTriggerMsg,} = this.props.ros
    const classOPtions = this.getClassOptions()
    const selectedClasses = this.state.selected_classes_list
    const NoneOption = <Option>None</Option>
    return (
      <Section title={"Targeting Controls *** Refresh page after AI running to update ***"}>


        <Columns>
        <Column>
        
        <Label title="Select Output Image"> 
              <Select
                id="select_output_image"
                onChange={(event) => onDropdownSelectedSendStr.bind(this)(event, this.props.targetingNamespace + "/set_output_image")}
                value={this.state.selected_output_image}
              >
                {this.state.output_image_options_list
                  ? createMenuListFromStrList(this.state.output_image_options_list, false, [],[],[])
                  : NoneOption}
              </Select>
            </Label>

            <Label title="Select Class Filter"> </Label>

                    <div onClick={this.toggleViewableTopics} style={{backgroundColor: Styles.vars.colors.grey0}}>
                      <Select style={{width: "10px"}}/>
                    </div>
                    <div hidden={this.state.viewableTopics === false}>
                    {classOPtions.map((Class) =>
                    <div onClick={this.onToggleClassSelection}
                      style={{
                        textAlign: "center",
                        padding: `${Styles.vars.spacing.xs}`,
                        color: Styles.vars.colors.black,
                        backgroundColor: (selectedClasses.includes(Class.props.value))? Styles.vars.colors.blue : Styles.vars.colors.grey0,
                        cursor: "pointer",
                        }}>
                        <body class_name ={Class} style={{color: Styles.vars.colors.black}}>{Class}</body>
                    </div>
                    )}
                    </div>



                    <Label title="Select Target Filter"> </Label>

                    <Select
                      id="select_target"
                      onChange={(event) => onDropdownSelectedSendStr.bind(this)(event, this.props.targetingNamespace + "/select_target")}
                      value={this.state.selected_target}
                    >
                      {this.state.available_targets_list
                        ? createMenuListFromStrList(this.state.available_targets_list, false, [],[],[])
                        : NoneOption}
                    </Select>


              </Column>
              <Column>

              <ButtonMenu>
            <Button onClick={() => sendTriggerMsg( this.props.targetingNamespace + "/reset_app")}>{"Reset App"}</Button>
          </ButtonMenu>

            <ButtonMenu>
              <Button onClick={() => sendTriggerMsg(this.props.targetingNamespace + "/save_config")}>{"Save Config"}</Button>
        </ButtonMenu>

        <ButtonMenu>
              <Button onClick={() => sendTriggerMsg( this.props.targetingNamespace + "/reset_config")}>{"Reset Config"}</Button>
        </ButtonMenu>


              </Column>
              </Columns>



      <Label title={"Sensor Vertical Degrees"}>
          <Input id="image_fov_vert_degs" 
            value={this.state.image_fov_vert_degs} 
            onChange={(event) => onUpdateSetStateValue.bind(this)(event,"image_fov_vert_degs")} 
            onKeyDown= {(event) => onEnterSendFloatValue.bind(this)(event,this.props.targetingNamespace + "/set_image_fov_vert")} />
        </Label>
           
        <Label title={"Sensor Horzontal Degrees"}>
          <Input id="image_fov_horz_degs" 
            value={this.state.image_fov_horz_degs} 
            onChange={(event) => onUpdateSetStateValue.bind(this)(event,"image_fov_vert_degs")} 
            onKeyDown= {(event) => onEnterSendFloatValue.bind(this)(event,this.props.targetingNamespace + "/set_image_fov_horz")} />
        </Label>

        <Label title={"Default Target Depth (m)"}>
          <Input id="default_target_depth_m" 
            value={this.state.default_target_depth_m} 
            onChange={(event) => onUpdateSetStateValue.bind(this)(event,"default_target_depth_m")} 
            onKeyDown= {(event) => onEnterSendFloatValue.bind(this)(event,this.props.targetingNamespace + "/set_default_target_detpth")} />
        </Label>

        <Label title={"Target Min Points"}>
          <Input id="target_min_points" 
            value={this.state.target_min_points} 
            onChange={(event) => onUpdateSetStateValue.bind(this)(event,"target_min_points")} 
            onKeyDown= {(event) => onEnterSendIntValue.bind(this)(event,this.props.targetingNamespace + "/set_target_min_points")} />
        </Label>

        <Label title={"Target Min Dist (m)"}>
          <Input id="target_min_dist_m" 
            value={this.state.target_min_dist_m} 
            onChange={(event) => onUpdateSetStateValue.bind(this)(event,"target_min_dist_m")} 
            onKeyDown= {(event) => onEnterSendFloatValue.bind(this)(event,this.props.targetingNamespace + "/set_target_min_dist_meters")} />
        </Label>

        <Label title={"Target Age Filter"}>
          <Input id="target_age_filter" 
            value={this.state.target_age_filter} 
            onChange={(event) => onUpdateSetStateValue.bind(this)(event,"target_age_filter")} 
            onKeyDown= {(event) => onEnterSendFloatValue.bind(this)(event,this.props.targetingNamespace + "/set_age_filter")} />
        </Label>

        <Label title={"Set Box Adjust %"}>
          <Input id="target_box_size_percent" 
            value={this.state.target_box_size_percent} 
            onChange={(event) => onUpdateSetStateValue.bind(this)(event,"target_box_size_percent")} 
            onKeyDown= {(event) => onEnterSendIntValue.bind(this)(event,this.props.targetingNamespace + "/set_target_box_size_percent")} />
        </Label>


        <SliderAdjustment
          title={"Target Min Pixel Ratio"}
          msgType={"std_msgs/float32"}
          adjustment={this.state.target_min_px_ratio}
          topic={this.props.targetingNamespace + "/set_target_min_px_ratio"}
          scaled={0.01}
          min={0}
          max={100}
          tooltip={""}
          unit={"%"}
      />






    <Columns>
    <Column>

    <Label title="Show 3D Transforms">
    <Toggle
      checked={this.state.showTransforms}
      onClick={this.onClickToggleShowTransforms}>
    </Toggle>
  </Label>

    </Column>
    <Column>

    </Column>
    </Columns>






    <div hidden={ this.state.showTransforms === false}>

      <Columns>
      <Column>

      <Label title={"X (m)"}>
            <Input
              value={this.state.transformTX}
              id="XTranslation"
              onChange= {(event) => onUpdateSetStateValue.bind(this)(event,"transformTX")}
              onKeyDown= {(event) => onEnterSetStateFloatValue.bind(this)(event,"transformTX")}
              style={{ width: "80%" }}
            />
          </Label>

          <Label title={"Y (m)"}>
            <Input
              value={this.state.transformTY}
              id="YTranslation"
              onChange= {(event) => onUpdateSetStateValue.bind(this)(event,"transformTY")}
              onKeyDown= {(event) => onEnterSetStateFloatValue.bind(this)(event,"transformTY")}
              style={{ width: "80%" }}
            />
          </Label>

          <Label title={"Z (m)"}>
            <Input
              value={this.state.transformTZ}
              id="ZTranslation"
              onChange= {(event) => onUpdateSetStateValue.bind(this)(event,"transformTZ")}
              onKeyDown= {(event) => onEnterSetStateFloatValue.bind(this)(event,"transformTZ")}
              style={{ width: "80%" }}
            />
          </Label>


          <ButtonMenu>
            <Button onClick={() => this.sendTransformUpdateMessage()}>{"Update Transform"}</Button>
          </ButtonMenu>


        </Column>
        <Column>

          <Label title={"Roll (deg)"}>
            <Input
              value={this.state.transformRX}
              id="XRotation"
              onChange= {(event) => onUpdateSetStateValue.bind(this)(event,"transformRX")}
              onKeyDown= {(event) => onEnterSetStateFloatValue.bind(this)(event,"transformRX")}
              style={{ width: "80%" }}
            />
          </Label>

          <Label title={"Pitch (deg)"}>
            <Input
              value={this.state.transformRY}
              id="YRotation"
              onChange= {(event) => onUpdateSetStateValue.bind(this)(event,"transformRY")}
              onKeyDown= {(event) => onEnterSetStateFloatValue.bind(this)(event,"transformRY")}
              style={{ width: "80%" }}
            />
          </Label>

              <Label title={"Yaw (deg)"}>
                <Input
                  value={this.state.transformRZ}
                  id="ZRotation"
                  onChange= {(event) => onUpdateSetStateValue.bind(this)(event,"transformRZ")}
                  onKeyDown= {(event) => onEnterSetStateFloatValue.bind(this)(event,"transformRZ")}
                  style={{ width: "80%" }}
                />
              </Label>


              <ButtonMenu>
            <Button onClick={() => this.sendClearTransformUpdateMessage()}>{"Clear Transform"}</Button>
          </ButtonMenu>



      </Column>
      </Columns>

      </div>



      </Section>

    
    )
  }

}
export default NepiAppAiTargetingControls
