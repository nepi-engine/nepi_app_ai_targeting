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
import Label from "./Label"
import { Column, Columns } from "./Columns"
import Styles from "./Styles"


import AppAiTargetingControls from "./NepiAppAiTargetingControls"
import AiDetectorMgr from "./NepiMgrAiDetector"
import CameraViewer from "./CameraViewer"
import NepiIFSaveData from "./Nepi_IF_SaveData"


import {round, convertStrToStrList, createMenuListFromStrList, onDropdownSelectedSendStr, onUpdateSetStateValue, onEnterSendFloatValue, onEnterSendIntValue, onEnterSetStateFloatValue} from "./Utilities"

@inject("ros")
@observer

// Component that contains the  Pointcloud App Viewer Controls
class NepiAppAiTargeting extends Component {
  constructor(props) {
    super(props)

    // these states track the values through  Status messages
    this.state = {
      appName: "app_ai_targeting",
      appNamespace: null,
    }
  
    this.getAppNamespace = this.getAppNamespace.bind(this)

  }

  getAppNamespace(){
    const { namespacePrefix, deviceId} = this.props.ros
    var appNamespace = null
    if (namespacePrefix !== null && deviceId !== null){
      appNamespace = "/" + namespacePrefix + "/" + deviceId + "/" + this.state.appName
    }
    return appNamespace
  }

  render() {
    const appNamespace = this.getAppNamespace()
    const imageNamespace = appNamespace + "/targeting_image"
    return (

      <Columns>
      <Column equalWidth={true}>

      <CameraViewer
        imageTopic={imageNamespace}
        title={this.state.selected_output_image}
        hideQualitySelector={false}
      />

      </Column>
      <Column>

      <AiDetectorMgr
              title={"Nepi_Mgr_AI_Detector"}
          />

      <AppAiTargetingControls
        appNamespace={appNamespace}
        title={"Nepi_App_AI_Targeting_Controls"}
      />

 
      <div hidden={appNamespace === null}>
        <NepiIFSaveData
          saveNamespace={appNamespace}
          title={"Nepi_IF_SaveData"}
        />
      </div>

      </Column>
      </Columns>



      )
    }  

}
export default NepiAppAiTargeting
