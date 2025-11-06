using System.Collections;
using UnityEngine;

public class RandomSphereSpawner : MonoBehaviour
{
    public GameObject spherePrefab; // Prefab for the sphere
    public Vector3[] spawnPositions; // Array of positions to spawn spheres
    public float sphereDuration = 2f; // Time before the sphere moves to another position
    public float totalDuration = 120f; // Total duration for the spawning process (2 minutes)
    public float sphereScale = 0.5f; // Scale factor for the sphere

    private GameObject currentSphere;
    private float elapsedTime = 0f;

    void Start()
    {
        StartCoroutine(SpawnSpheres());
    }

    IEnumerator SpawnSpheres()
    {
        // Instantiate the sphere once
        currentSphere = Instantiate(spherePrefab);
        currentSphere.transform.localScale = new Vector3(sphereScale, sphereScale, sphereScale); // Scale down the sphere

        while (elapsedTime < totalDuration) // Loop for 2 minutes
        {
            // Randomly select a position from the array
            Vector3 randomPosition = spawnPositions[Random.Range(0, spawnPositions.Length)];

            // Move the sphere to the random position
            currentSphere.transform.position = randomPosition;

            // Ensure the sphere is active
            if (!currentSphere.activeInHierarchy)
            {
                currentSphere.SetActive(true);
            }

            // Wait for the specified duration
            yield return new WaitForSeconds(sphereDuration);

            // Update the elapsed time
            elapsedTime += sphereDuration;
            // Print that the sphere shape is changed
            Debug.Log("Sphere position is changed");
        }

        // After 2 minutes, deactivate the sphere
        currentSphere.SetActive(false);
    }
}