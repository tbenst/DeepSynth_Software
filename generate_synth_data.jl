tifdirs = ["/mnt/deissero/users/tyler/b115/2020-10-05_elavl3-chrmine-kv2.1_h2b-6s/ZSeries-25X-default-exponential-zpower-64avg-034"]

out_dir = "./Spcyclegan/datasets/zfish-b115-25x-0.9x-v3"
ENV["DISPLAY"] = "localhost:11.0"

using LazySets, LinearAlgebra, ImageView, Images, ImageMagick, TiledIteration,
    StatsBase, ImageCore, Distributions
# REAL DATA
for (i,dir) in enumerate(tifdirs)    
    tifs = filter(x->occursin(r".*_Ch3_\d+.ome.tif$",x),readdir(dir))
    global tifs = joinpath.(dir, tifs)
    zstack = Gray.(cat([ImageMagick.load(tif) for tif in tifs]...,dims=3))
    global zseries = imadjustintensity(zstack)
    imshow(zstack)
    # warning: this will change in 0.1.1
    # zstack = adjust_histogram(zstack, Equalization())
    # zstack = imadjustintensity(zstack)
    maxval = percentile(zstack[:],99)
    minval = percentile(zstack[:],1)
    # do some kind of z-dependent correction...?
    # or maybe the network should learn this..?
    # zmaxes = mapslices(a->percentile(a,99),reshape(zstack,:,size(zstack,3)),dims=1)
    # startZ = 130 # hab
    global startZ = 50 #optic tectum
    global endZ = 120 #optic tectum
    for (z,tif) in zip(startZ:endZ, tifs[startZ:endZ])
        zplane = match(r"(\d+).ome.tif", tif)[1]
        raw_im = zstack[:,:,z]
        # im = map(scaleminmax(minval, maxval), raw_im)
        im = imadjustintensity(raw_im)
        H, W = size(im)
        # startH = Int(floor(H//2))
        # startW = Int(floor(W//4))
        startH = 280 # optic tectum
        startW = 475 # optic tectum
        # startH = 850 # hab
        # startW = 270 # hab
        save(joinpath(out_dir, "trainB/z$zplane.png"),
            raw_im[startH:startH+64,startW:startW+64])
        save(joinpath(out_dir, "trainB_adj/z$zplane.png"),
        im[startH:startH+64,startW:startW+64])
        # TODO: how to train on all patches..?
        # for (j, tileaxs) in enumerate(TileIterator(axes(im), (64,64)))
        #     if minimum(getindex.(size.(tileaxs),1)) < 64
        #         continue
        #     else
        #         @show "vol$i_"
        # end
    end
end
# SYNTHETIC DATA
# ex_im = imadjustintensity(ImageMagick.load(tifs[100]))
# imgH, imgW = size(ex_im)
# imgD = endZ - startZ

function sample_ellipsoid(imgH, imgW, imgD;
    height=(8,20), width=(8,20), depth=(4,10))
    h = float64(rand(height[1]:height[2]))
    w = float64(rand(width[1]:width[2]))
    d = float64(rand(depth[1]:depth[2]))
    x = float64(rand(1:imgW))
    y = float64(rand(1:imgH))
    z = float64(rand(1:imgD))
    return Ellipsoid([y,x,z], Diagonal([h^2, w^2, d^2]))
end

function sample_vol_with_ellipsoid(imgH, imgW, imgD;
    height=(8,20), width=(8,20), depth=(4,10))
    im = zeros(Bool, imgH, imgW, imgD)
    E = sample_ellipsoid(imgH, imgW, imgD; height, width, depth)
    for c in CartesianIndices(im)
        if collect(Float64.(Tuple(c))) ∈ E
            im[c] = true
        end
    end
    im
end

function sample_synthetic_vol(imgH, imgW, imgD;
    height=(8,20), width=(8,20), depth=(4,10), maxInvalidSamples=10)
    nInvalidSamples = 0
    vol = sample_vol_with_ellipsoid(64, 64, 64; height=height, width=width, depth=depth)
    while nInvalidSamples < maxInvalidSamples
        new_vol = sample_vol_with_ellipsoid(64, 64, 64;
            height=height, width=width, depth=depth)
        if sum(vol .& new_vol) <=5
            vol = new_vol .| vol
        else
            nInvalidSamples += 1
        end
    end
    vol
end

# TRAIN
vol = sample_synthetic_vol(64, 64, 64; height=(3,6), width=(3,6), depth=(3,6),
    maxInvalidSamples=100)
# imshow(vol)
#
for (z, im) in enumerate(eachslice(vol,dims=1))
    save(joinpath(out_dir, "trainA/1_$z.png"), im)
end

# TEST
vol = sample_synthetic_vol(64, 64, 64; height=(5,8), width=(5,8), depth=(5,8),
    maxInvalidSamples=100)

for (z, im) in enumerate(eachslice(vol,dims=1))
    save(joinpath(out_dir, "testA/1_$z.png"), im)
end

"done!"
# im = sample_vol_with_ellipsoid(64, 64, 64; height=(5,8), width=(5,8), depth=(5,8))
# imshow(im)