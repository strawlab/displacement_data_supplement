run("Subtract Background...", "rolling=10 light");
run("Brightness/Contrast...");
setMinAndMax(195, 256);
call("ij.ImagePlus.setDefault16bitRange", 8);

