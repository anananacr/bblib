
def select_best_center(coordinates: list) -> list:
    fwhm_summary = []
    fwhm_over_radius_summary = []
    r_squared_summary = []
    good_coordinates = []
    directions_summary = []
    movement = ["+x", "-x", "+y", "-y", "0"]
    for idx, i in enumerate(coordinates):
        # Update center for pf8 with the last calculated center
        # print(pf8_info)
        pf8_info.modify_radius(i[0], i[1])
        pf8_info._bad_pixel_map = mask

        # Update geom and recorrect polarization
        updated_geom = (
            f"{args.geom[:-5]}_{label}_{frame_number}_fwhm_{i[0]}_{i[1]}.geom"
        )
        cmd = f"cp {args.geom} {updated_geom}"
        sub.call(cmd, shell=True)
        update_corner_in_geom(updated_geom, i[0], i[1])
        x_map, y_map, det_dict = gf.pixel_maps_from_geometry_file(
            updated_geom, return_dict=True
        )
        corrected_data, _ = correct_polarization(x_map, y_map, clen_v, data, mask=mask)

        # Find Bragg peaks list with pf8
        pf8 = PF8(pf8_info)
        peak_list = pf8.get_peaks_pf8(data=corrected_data)
        indices = (
            np.array(peak_list["ss"], dtype=int),
            np.array(peak_list["fs"], dtype=int),
        )

        only_peaks_mask = mask_peaks(mask, indices, bragg=0)
        pf8_mask = only_peaks_mask * mask

        x, y = azimuthal_average(corrected_data, center=i, mask=pf8_mask)

        ## Define background peak region
        x_min = 200
        x_max = 350
        x = x[x_min:x_max]
        y = y[x_min:x_max]

        ## Estimation of initial parameters
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))

        popt, pcov = curve_fit(gaussian, x, y, p0=[max(y), mean, sigma])
        residuals = y - gaussian(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        ## Calculation of FWHM
        fwhm = popt[2] * math.sqrt(8 * np.log(2))

        ## Divide by radius of the peak to get shasrpness ratio
        fwhm_over_radius = fwhm / popt[1]
        # print(r_squared)
        if r_squared > 0:
            good_coordinates.append(i)
            fwhm_summary.append(fwhm)
            fwhm_over_radius_summary.append(fwhm_over_radius)
            r_squared_summary.append(r_squared)
            directions_summary.append(movement[idx])
    #print(good_coordinates)
    #print(np.array(fwhm_summary))
    sorted_candidates = sorted(
        list(
            zip(fwhm_summary, good_coordinates, r_squared_summary, directions_summary)
        ),
        key=lambda x: x[0],
    )
    fwhm_summary, good_coordinates, r_squared_summary, match_directions = zip(
        *sorted_candidates
    )
    if match_directions[0][-1] == "y" and match_directions[1][-1] == "x":
        xc = good_coordinates[1][0]
        yc = good_coordinates[0][1]
        results = calculate_fwhm((xc, yc))
        combined_fwhm = results["fwhm"]
        combined_fwhm_r_sq = results["r_squared"]
        if combined_fwhm < fwhm_summary[0]:
            best_center = [xc, yc]
            best_fwhm = combined_fwhm
            best_fwhm_r_squared = combined_fwhm_r_sq
        else:
            best_center = list(good_coordinates[0])
            best_fwhm = fwhm_summary[0]
            best_fwhm_r_squared = r_squared_summary[0]
    elif match_directions[0][-1] == "x" and match_directions[1][-1] == "y":
        xc = good_coordinates[0][0]
        yc = good_coordinates[1][1]
        results = calculate_fwhm((xc, yc))
        combined_fwhm = results["fwhm"]
        combined_fwhm_r_sq = results["r_squared"]
        if combined_fwhm < fwhm_summary[0]:
            best_center = [xc, yc]
            best_fwhm = combined_fwhm
            best_fwhm_r_squared = combined_fwhm_r_sq
        else:
            best_center = list(good_coordinates[0])
            best_fwhm = fwhm_summary[0]
            best_fwhm_r_squared = r_squared_summary[0]
    else:
        best_center = list(good_coordinates[0])
        best_fwhm = fwhm_summary[0]
        best_fwhm_r_squared = r_squared_summary[0]

    # print(best_center)
    return (best_center, best_fwhm, best_fwhm_r_squared)


def direct_search_fwhm(initial_center: list) -> Dict[str, int]:

    step = 15
    last_center = initial_center
    next_center = initial_center
    center_pos_summary = [next_center]
    r_squared_summary = []
    fwhm_summary = []
    distance_x = 1
    distance_y = 1

    max_iter = 100
    n_iter = 0

    while n_iter < max_iter and step>0:
        # and distance_x > 0.5 and distance_y>0.5

        coordinates = [
            (next_center[0] + step, next_center[1]),
            (next_center[0] - step, next_center[1]),
            (next_center[0], next_center[1] + step),
            (next_center[0], next_center[1] - step),
            (next_center[0], next_center[1] )
        ]
        # print(coordinates)
        next_center, fwhm, r_squared = select_best_center(coordinates)
        center_pos_summary.append(next_center)
        fwhm_summary.append(round(fwhm, 10))
        r_squared_summary.append(round(r_squared, 10))
        distance_x, distance_y = (
            next_center[0] - last_center[0],
            next_center[1] - last_center[1],
        )
        if next_center==last_center and next_center==center_pos_summary[-2]:
            step-=1
        if next_center!=last_center:
            step=15
        last_center = next_center.copy()
        # print(distance_x,distance_y)   
        n_iter += 1

    fwhm_summary=np.array(fwhm_summary).reshape((n_iter,))
    
    final_center =  next_center

    return {
        "xc": final_center[0],
        "yc": final_center[1],
        "center_pos_summary": center_pos_summary,
        "fwhm_summary": fwhm_summary,
        "r_squared_summary": np.array(r_squared_summary).reshape((n_iter,)),
    }
