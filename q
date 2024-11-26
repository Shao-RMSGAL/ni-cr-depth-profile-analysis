[1mdiff --git a/concentration_analysis.py b/concentration_analysis.py[m
[1mindex 303ed33..caa131d 100644[m
[1m--- a/concentration_analysis.py[m
[1m+++ b/concentration_analysis.py[m
[36m@@ -174,12 +174,12 @@[m [mdef get_conc_profiles(cr_binary, ni_binary, mo_binary, fe_binary, segments):[m
         cr_segment = cr_binary[cr_prev_index:cr_next_index, :][m
         ni_segment = ni_binary[ni_prev_index:ni_next_index, :][m
         mo_segment = mo_binary[mo_prev_index:mo_next_index, :][m
[31m-        fe_segment = ni_binary[fe_prev_index:fe_next_index, :][m
[32m+[m[32m        fe_segment = fe_binary[fe_prev_index:fe_next_index, :][m
         # Count white pixels in the segment[m
[31m-        cr_profile[i] = np.sum(cr_segment == 255) # / cr_segment.size[m
[31m-        ni_profile[i] = np.sum(ni_segment == 255) # / ni_segment.size[m
[31m-        mo_profile[i] = np.sum(mo_segment == 255) # / mo_segment.size[m
[31m-        fe_profile[i] = np.sum(fe_segment == 255) # / fe_segment.size[m
[32m+[m[32m        cr_profile[i] = np.sum(cr_segment == 255)  # / cr_segment.size[m
[32m+[m[32m        ni_profile[i] = np.sum(ni_segment == 255)  # / ni_segment.size[m
[32m+[m[32m        mo_profile[i] = np.sum(mo_segment == 255)  # / mo_segment.size[m
[32m+[m[32m        fe_profile[i] = np.sum(fe_segment == 255)  # / fe_segment.size[m
     return cr_profile, ni_profile, mo_profile, fe_profile[m
 [m
 [m
[36m@@ -236,15 +236,17 @@[m [mfor class_name in class_path:[m
         _, mo_binary = cv.threshold(mo_img, mo_threshold, 255, cv.THRESH_BINARY)[m
         _, fe_binary = cv.threshold(fe_img, fe_threshold, 255, cv.THRESH_BINARY)[m
 [m
[32m+[m
         # Number of segments to use when profiling[m
         segments = 100[m
         cr_profile, ni_profile, mo_profile, fe_profile = get_conc_profiles([m
                 cr_binary, ni_binary, mo_binary, fe_binary, segments)[m
 [m
[32m+[m[41m        [m
         path = os.path.join(output_dir, class_name, material)[m
        [m
         write_images([cr_img, ni_img, mo_img, fe_img], elements, path)[m
[31m-        plot_profiles(cr_profile, ni_profile, mo_profile, fe_profile, height, [m
[32m+[m[32m        plot_profiles(cr_profile, ni_profile, mo_profile, fe_profile, height,[m
                                                                         path)[m
         show_binary_images(cr_binary, ni_binary, mo_binary, fe_binary, width, height, path)[m
         save_data(segments, height, cr_profile, ni_profile, mo_profile, [m
[1mdiff --git a/images/1953 45000 x SI EDS-HAADF-Cr-at.bmp b/images/1953 45000 x SI EDS-HAADF-Cr-at.bmp[m
[1mdeleted file mode 100644[m
[1mindex e8b9a89..0000000[m
Binary files a/images/1953 45000 x SI EDS-HAADF-Cr-at.bmp and /dev/null differ
[1mdiff --git a/images/1953 45000 x SI EDS-HAADF-Ni-at.bmp b/images/1953 45000 x SI EDS-HAADF-Ni-at.bmp[m
[1mdeleted file mode 100644[m
[1mindex 92bcd32..0000000[m
Binary files a/images/1953 45000 x SI EDS-HAADF-Ni-at.bmp and /dev/null differ
