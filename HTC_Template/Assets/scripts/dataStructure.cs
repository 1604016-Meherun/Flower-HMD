using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System;
using UnityEngine.SceneManagement;

public class dataStructure : MonoBehaviour
{

    [System.Serializable]
    public class ObjectData
    {
        public float posX, posY, posZ;
        public float rotX, rotY, rotZ, rotW;
        public float dirX, dirY, dirZ;

        public ObjectData(Transform transform)
        {
            Vector3 position = transform.position;
            Quaternion rotation = transform.rotation;
            Vector3 direction = transform.forward;

            posX = position.x;
            posY = position.y;
            posZ = position.z;

            rotX = rotation.x;
            rotY = rotation.y;
            rotZ = rotation.z;
            rotW = rotation.w;

            dirX = direction.x;
            dirY = direction.y;
            dirZ = direction.z;
        }

        public string ToCsvString()
        {
            return $"{posX},{posY},{posZ},{rotX},{rotY},{rotZ},{rotW},{dirX},{dirY},{dirZ}";
        }
    }

    [System.Serializable]
    public class FrameData
    {
        public string sceneName;
        public float elapsedTime;
        public ObjectData leftEye, rightEye, centerEye;

        public FrameData(string scene, float time, ObjectData left, ObjectData right, ObjectData center)
        {
            sceneName = scene;
            elapsedTime = time;
            leftEye = left;
            rightEye = right;
            centerEye = center;
        }

        public string ToCsvString()
        {
            return $"{sceneName},{elapsedTime:F4},{leftEye.ToCsvString()},{rightEye.ToCsvString()},{centerEye.ToCsvString()}";
        }
    }
}