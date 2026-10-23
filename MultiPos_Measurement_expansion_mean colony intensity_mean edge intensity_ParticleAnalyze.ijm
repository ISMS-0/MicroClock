// ================================================================
// Batch processing of multiple Pos folders using the same ROI set
// ================================================================

// ---- USER SETTINGS ----
inputfolder        = "E:/BKK_8.2/";
positionsToProcess = newArray(7, 8);   // <--- choose which Pos folders to run
roiFolder          = inputfolder + "Rois/";

// ---- Parameters to set for each experiment ----
max_contrast = 10228;
threshold = 11937;
edgeWidth = 4;
edgeFromCumulativeMask = true;
saveMasks = true;

// ---- Loop over selected Pos folders ----
for (p = 0; p < positionsToProcess.length; p++) {

    posNum = positionsToProcess[p];
    input  = inputfolder + "pos" + posNum + "/";
    outputFolder = inputfolder + "Output" + posNum + "/";

    print("\\n==============================");
    print("Processing Pos " + posNum);
    print("==============================");

    File.makeDirectory(outputFolder);

    // Output CSV files for this Pos
    areaOutput  = outputFolder + "colony_area_results.csv";
    edgeOutput  = outputFolder + "edge_bioluminescence_results.csv";
    wholeOutput = outputFolder + "whole_colony_bioluminescence_results.csv";

    // Sorted input image and ROI lists
    imgList = getFileList(input);
    Array.sort(imgList);

    roiList = getFileList(roiFolder);
    Array.sort(roiList);

    // Create tables for this Pos
    Table.create("ColonyAreaResults");
    Table.create("EdgeBioluminescenceResults");
    Table.create("WholeColonyBioluminescenceResults");

    // ---- ROI LOOP ----
    for (r = 0; r < roiList.length; r++) {

        roiFile = roiList[r];
        if (!endsWith(roiFile, ".roi"))
            continue;

        roiName = substring(roiFile, 0, lengthOf(roiFile) - 4);
        print("Processing ROI: " + roiName);

        roiManager("Reset");
        roiManager("Open", roiFolder + roiFile);

        roiOutput = outputFolder + roiName + "/";
        File.makeDirectory(roiOutput);

        firstFrame = true;
        cumID = -1;
        frame = 0;

        // ---- IMAGE LOOP ----
        for (i = 0; i < imgList.length; i++) {

            file = imgList[i];
            if (!endsWith(file, ".tif")) continue;
            if (File.isDirectory(input + file)) continue;

            print("   " + file);

            // Crop to ROI
            open(input + file);
            rawID = getImageID();
            roiManager("Select", 0);
            run("Duplicate...", "title=roiImage");
            roiID = getImageID();
            selectImage(rawID);
            close();

            // Duplicate raw for intensity measurement
            selectImage(roiID);
            run("Duplicate...", "title=rawIntensityImage");
            rawIntensityID = getImageID();

            // Create current mask
            selectImage(roiID);
            run("Duplicate...", "title=currentMask");
            currentMaskID = getImageID();
            selectImage(roiID);
            close();

            selectImage(currentMaskID);
            setMinAndMax(0, max_contrast);
            run("Apply LUT");
            setThreshold(threshold, 65535);
            setOption("BlackBackground", true);
            run("Convert to Mask");

            // Remove hot pixels
            run("Minimum...", "radius=2");
            run("Maximum...", "radius=2");

            currentMaskID = getImageID();

            // Duplicate for cumulative update
            selectImage(currentMaskID);
            run("Duplicate...", "title=cumulativeUpdate");
            updateID = getImageID();

            // Build cumulative mask
            if (firstFrame) {
                cumID = updateID;
                firstFrame = false;
            } else {
                selectImage(cumID);
                cumTitle = getTitle();
                selectImage(updateID);
                updateTitle = getTitle();
                imageCalculator("OR create", cumTitle, updateTitle);
                newCumID = getImageID();

                selectImage(cumID); close();
                selectImage(updateID); close();
                cumID = newCumID;
            }

            // Measure cumulative area + whole colony intensity
            selectImage(cumID);
            setThreshold(1, 255);
            run("Create Selection");

            if (selectionType() != -1) {
                getStatistics(cumulativeArea, cumulativeMaskMean);

                roiManager("Add");
                cumRoiIndex = roiManager("count") - 1;

                selectImage(rawIntensityID);
                roiManager("Select", cumRoiIndex);
                getStatistics(wholeArea, wholeColonyMean);

                roiManager("Select", cumRoiIndex);
                roiManager("Delete");
            } else {
                cumulativeArea = 0;
                wholeColonyMean = 0;
            }

            selectImage(rawIntensityID); run("Select None");
            selectImage(cumID); run("Select None");

            // Inner edge mask
            if (edgeFromCumulativeMask)
                selectImage(cumID);
            else
                selectImage(currentMaskID);

            run("Duplicate...", "title=edgeSourceMask");
            edgeSourceID = getImageID();

            selectImage(edgeSourceID);
            run("Duplicate...", "title=erodedMask");
            erodedID = getImageID();
            setOption("BlackBackground", true);

            for (e = 0; e < edgeWidth; e++)
                run("Erode");

            selectImage(edgeSourceID);
            edgeSourceTitle = getTitle();
            selectImage(erodedID);
            erodedTitle = getTitle();
            imageCalculator("Subtract create", edgeSourceTitle, erodedTitle);
            innerEdgeID = getImageID();

            selectImage(innerEdgeID);
            setThreshold(1, 255);
            run("Create Selection");

            if (selectionType() != -1) {
                roiManager("Add");
                edgeRoiIndex = roiManager("count") - 1;

                selectImage(rawIntensityID);
                roiManager("Select", edgeRoiIndex);
                getStatistics(edgeArea, edgeMean);

                roiManager("Select", edgeRoiIndex);
                roiManager("Delete");
            } else {
                edgeMean = 0;
            }

            run("Select None");

            // Store results
            Table.set("Frame", frame, frame + 1, "ColonyAreaResults");
            Table.set("Filename", frame, file, "ColonyAreaResults");
            Table.set(roiName + "_CumulativeArea", frame, cumulativeArea, "ColonyAreaResults");

            Table.set("Frame", frame, frame + 1, "EdgeBioluminescenceResults");
            Table.set("Filename", frame, file, "EdgeBioluminescenceResults");
            Table.set(roiName + "_EdgeMean", frame, edgeMean, "EdgeBioluminescenceResults");

            Table.set("Frame", frame, frame + 1, "WholeColonyBioluminescenceResults");
            Table.set("Filename", frame, file, "WholeColonyBioluminescenceResults");
            Table.set(roiName + "_WholeMean", frame, wholeColonyMean, "WholeColonyBioluminescenceResults");

            base = substring(file, 0, lengthOf(file) - 4);

            if (saveMasks) {
                selectImage(cumID);
                saveAs("Tiff", roiOutput + "cumMask_" + base + ".tif");
                selectImage(innerEdgeID);
                saveAs("Tiff", roiOutput + "innerEdgeMask_" + base + ".tif");
            }

            // Cleanup
            selectImage(innerEdgeID); close();
            selectImage(erodedID); close();
            selectImage(edgeSourceID); close();
            selectImage(currentMaskID); close();
            selectImage(rawIntensityID); close();

            frame++;
        }

        if (cumID != -1) {
            selectImage(cumID);
            close();
        }
    }

    // Save tables for this Pos
    selectWindow("ColonyAreaResults");
    Table.save(areaOutput);
    selectWindow("EdgeBioluminescenceResults");
    Table.save(edgeOutput);
    selectWindow("WholeColonyBioluminescenceResults");
    Table.save(wholeOutput);

    print("Finished Pos " + posNum);
    print("Saved:");
    print("   " + areaOutput);
    print("   " + edgeOutput);
    print("   " + wholeOutput);
}

print("\\nAll requested Pos folders processed.");
