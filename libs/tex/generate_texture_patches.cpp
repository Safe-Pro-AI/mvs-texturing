/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <set>
#include <list>

#include <util/timer.h>
#include <mve/image_tools.h>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>

#include "texturing.h"

TEX_NAMESPACE_BEGIN

#define MAX_HOLE_NUM_FACES 1000
#define MAX_HOLE_PATCH_SIZE 1000

template <typename T>
T clamp_nan_low(T const & v, T const & lo, T const & hi) {
    return (v > lo) ? ((v < hi) ? v : hi) : lo;
}

template <typename T>
T clamp_nan_hi(T const & v, T const & lo, T const & hi) {
    return (v < hi) ? ((v > lo) ? v : lo) : hi;
}

template <typename T>
T clamp(T const & v, T const & lo, T const & hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

void merge_vertex_projection_infos(std::vector<std::vector<VertexProjectionInfo> > * vertex_projection_infos) {
    /* Merge vertex infos within the same texture patch. */
    #pragma omp parallel for
#if !defined(_MSC_VER)
    for (std::size_t i = 0; i < vertex_projection_infos->size(); ++i) {
#else
    for (std::int64_t i = 0; i < vertex_projection_infos->size(); ++i) {
#endif
        std::vector<VertexProjectionInfo> & infos = vertex_projection_infos->at(i);

        std::map<std::size_t, VertexProjectionInfo> info_map;
        std::map<std::size_t, VertexProjectionInfo>::iterator it;

        for (VertexProjectionInfo const & info : infos) {
            std::size_t texture_patch_id = info.texture_patch_id;
            if((it = info_map.find(texture_patch_id)) == info_map.end()) {
                info_map[texture_patch_id] = info;
            } else {
                it->second.faces.insert(it->second.faces.end(),
                    info.faces.begin(), info.faces.end());
            }
        }

        infos.clear();
        infos.reserve(info_map.size());
        for (it = info_map.begin(); it != info_map.end(); ++it) {
            infos.push_back(it->second);
        }
    }
}

/** Struct representing a TexturePatch candidate
 * - final texture patches are obtained by merging candiates. */
struct TexturePatchCandidate {
    Rect<int> bounding_box;
    TexturePatch::Ptr texture_patch;
};

/** Create a TexturePatchCandidate by calculating the faces' bounding box
 * projected into the view,
 *  relative texture coordinates and extacting the texture views relevant part
 */
TexturePatchCandidate
generate_candidate(int label, TextureView const & texture_view,
    std::vector<std::size_t> const & faces, mve::TriangleMesh::ConstPtr mesh,
    Settings const & settings)
{
    mve::ImageBase::Ptr view_image = texture_view.get_image();
    const int img_w = view_image->width();
    const int img_h = view_image->height();

    // Empty bbox to start
    int min_x = img_w;
    int min_y = img_h;
    int max_x = 0;
    int max_y = 0;

    mve::TriangleMesh::FaceList const & mesh_faces = mesh->get_faces();
    mve::TriangleMesh::VertexList const & vertices = mesh->get_vertices();

    std::vector<math::Vec2f> texcoords;
    texcoords.reserve(faces.size() * 3);

    auto sanitize_uv = [img_w, img_h](float &u, float &v) {
        // Treat non-finite or utterly insane values as invalid
        if (!std::isfinite(u) || !std::isfinite(v) ||
            u < -1e6f || v < -1e6f ||
            u > img_w + 1e6f || v > img_h + 1e6f)
        {
            u = (img_w - 1) * 0.5f;
            v = (img_h - 1) * 0.5f;
        }

        // Clamp to image bounds
        if (u < 0.0f) u = 0.0f;
        if (v < 0.0f) v = 0.0f;
        float max_u = static_cast<float>(img_w - 1);
        float max_v = static_cast<float>(img_h - 1);
        if (u > max_u) u = max_u;
        if (v > max_v) v = max_v;
    };

    // --- Gather texcoords and global bbox in source image space ---
    for (std::size_t i = 0; i < faces.size(); ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            math::Vec3f vertex = vertices[mesh_faces[faces[i] * 3 + j]];
            math::Vec2f pixel  = texture_view.get_pixel_coords(vertex);

            float u = pixel[0];
            float v = pixel[1];
            sanitize_uv(u, v);

            math::Vec2f uv(u, v);
            texcoords.push_back(uv);

            int fx = static_cast<int>(std::floor(u));
            int fy = static_cast<int>(std::floor(v));
            int cx = static_cast<int>(std::ceil(u));
            int cy = static_cast<int>(std::ceil(v));

            if (fx < min_x) min_x = fx;
            if (fy < min_y) min_y = fy;
            if (cx > max_x) max_x = cx;
            if (cy > max_y) max_y = cy;
        }
    }

    // If everything went sideways, make a 1x1 patch at the image center
    if (texcoords.empty() || min_x > max_x || min_y > max_y) {
        int cx = img_w  > 0 ? (img_w  - 1) / 2 : 0;
        int cy = img_h > 0 ? (img_h - 1) / 2 : 0;

        min_x = max_x = cx;
        min_y = max_y = cy;

        texcoords.clear();
        texcoords.reserve(faces.size() * 3);
        for (std::size_t i = 0; i < faces.size() * 3; ++i) {
            texcoords.emplace_back(static_cast<float>(cx),
                                   static_cast<float>(cy));
        }
    }

    // Grow bbox by border in source-image space
    min_x -= texture_patch_border;
    min_y -= texture_patch_border;
    max_x += texture_patch_border;
    max_y += texture_patch_border;

    // Clamp bbox to the image bounds
    min_x = std::max(0, std::min(min_x, img_w  - 1));
    min_y = std::max(0, std::min(min_y, img_h - 1));
    max_x = std::max(0, std::min(max_x, img_w  - 1));
    max_y = std::max(0, std::min(max_y, img_h - 1));

    if (min_x > max_x) std::swap(min_x, max_x);
    if (min_y > max_y) std::swap(min_y, max_y);

    int width  = max_x - min_x + 1;
    int height = max_y - min_y + 1;

    if (width  <= 0) width  = 1;
    if (height <= 0) height = 1;

    // --- Rebase texcoords into patch-local coordinates (0..width, 0..height) ---
    math::Vec2f origin(static_cast<float>(min_x), static_cast<float>(min_y));
    for (std::size_t i = 0; i < texcoords.size(); ++i) {
        texcoords[i] -= origin;
    }

    // --- Build the patch image from the source view ---
    mve::FloatImage::Ptr image;

    if (view_image->get_type() == mve::IMAGE_TYPE_FLOAT) {
        mve::FloatImage::Ptr float_image = texture_view.get_image<float>();
        const float fill_color[3] = {
            std::numeric_limits<float>::max(),
            0.0f,
            std::numeric_limits<float>::max()
        };
        image = mve::image::crop(
            float_image, width, height, min_x, min_y, fill_color);
    }
    else if (view_image->get_type() == mve::IMAGE_TYPE_UINT16) {
        mve::RawImage::Ptr raw_image = texture_view.get_image<uint16_t>();
        const uint16_t fill_color[3] = {
            std::numeric_limits<uint16_t>::max(),
            static_cast<uint16_t>(0),
            std::numeric_limits<uint16_t>::max()
        };
        mve::RawImage::Ptr cropped = mve::image::crop(
            raw_image, width, height, min_x, min_y, fill_color);
        image = mve::image::raw_to_float_image(cropped);
    }
    else {
        mve::ByteImage::Ptr byte_image = texture_view.get_image<uint8_t>();
        const uint8_t fill_color[3] = {
            std::numeric_limits<uint8_t>::max(),
            static_cast<uint8_t>(0),
            std::numeric_limits<uint8_t>::max()
        };
        mve::ByteImage::Ptr cropped = mve::image::crop(
            byte_image, width, height, min_x, min_y, fill_color);
        image = mve::image::byte_to_float_image(cropped);
    }

    if (settings.tone_mapping == TONE_MAPPING_GAMMA) {
        mve::image::gamma_correct(image, 2.2f);
    }

    // IMPORTANT: make the ROI purely local and consistent with image size.
    Rect<int> roi(0, 0, width - 1, height - 1);

    TexturePatchCandidate texture_patch_candidate =
        { roi, TexturePatch::create(label, faces, texcoords, image) };

    return texture_patch_candidate;
}



bool fill_hole(std::vector<std::size_t> const & hole, UniGraph const & graph,
    mve::TriangleMesh::ConstPtr mesh, mve::MeshInfo const & mesh_info,
    std::vector<std::vector<VertexProjectionInfo> > * vertex_projection_infos,
    std::vector<TexturePatch::Ptr> * texture_patches) {

    mve::TriangleMesh::FaceList const & mesh_faces = mesh->get_faces();
    mve::TriangleMesh::VertexList const & vertices = mesh->get_vertices();

    std::map<std::size_t, std::set<std::size_t> > tmp;
    for (std::size_t const face_id : hole) {
        std::size_t const v0 = mesh_faces[face_id * 3 + 0];
        std::size_t const v1 = mesh_faces[face_id * 3 + 1];
        std::size_t const v2 = mesh_faces[face_id * 3 + 2];

        tmp[v0].insert(face_id);
        tmp[v1].insert(face_id);
        tmp[v2].insert(face_id);
    }

    std::size_t const num_vertices = tmp.size();
    /* Only fill small holes. */
    if (num_vertices > MAX_HOLE_NUM_FACES) return false;

    /* Calculate 2D parameterization using the technique from libremesh/patch2d,
     * which was published as source code accompanying the following paper:
     *
     * Isotropic Surface Remeshing
     * Simon Fuhrmann, Jens Ackermann, Thomas Kalbe, Michael Goesele
     */

    std::size_t seed = -1;
    std::vector<bool> is_border(num_vertices, false);
    std::vector<std::vector<std::size_t> > adj_verts_via_border(num_vertices);
    /* Index structures to map from local <-> global vertex id. */
    std::map<std::size_t, std::size_t> g2l;
    std::vector<std::size_t> l2g(num_vertices);
    /* Index structure to determine column in matrix/vector. */
    std::vector<std::size_t> idx(num_vertices);

    std::size_t num_border_vertices = 0;

    bool disk_topology = true;
    std::map<std::size_t, std::set<std::size_t> >::iterator it = tmp.begin();
    for (std::size_t j = 0; j < num_vertices; ++j, ++it) {
        std::size_t vertex_id = it->first;
        g2l[vertex_id] = j;
        l2g[j] = vertex_id;

        /* Check topology in original mesh. */
        if (mesh_info[vertex_id].vclass != mve::MeshInfo::VERTEX_CLASS_SIMPLE) {
            /* Complex/Border vertex in original mesh */
            disk_topology = false;
            break;
        }

        /* Check new topology and determine if vertex is now at the border. */
        std::vector<std::size_t> const & adj_faces = mesh_info[vertex_id].faces;
        std::set<std::size_t> const & adj_hole_faces = it->second;
        std::vector<std::pair<std::size_t, std::size_t> > fan;
        for (std::size_t k = 0; k < adj_faces.size(); ++k) {
            std::size_t adj_face = adj_faces[k];
            if (graph.get_label(adj_faces[k]) == 0 &&
                adj_hole_faces.find(adj_face) != adj_hole_faces.end()) {
                std::size_t curr = adj_faces[k];
                std::size_t next = adj_faces[(k + 1) % adj_faces.size()];
                std::pair<std::size_t, std::size_t> pair(curr, next);
                fan.push_back(pair);
            }
        }

        std::size_t gaps = 0;
        for (std::size_t k = 0; k < fan.size(); k++) {
            std::size_t curr = fan[k].first;
            std::size_t next = fan[(k + 1) % fan.size()].first;
            if (fan[k].second != next) {
                ++gaps;

                for (std::size_t l = 0; l < 3; ++l) {
                    if(mesh_faces[curr * 3 + l] == vertex_id) {
                        std::size_t second = mesh_faces[curr * 3 + (l + 2) % 3];
                        adj_verts_via_border[j].push_back(second);
                    }
                    if(mesh_faces[next * 3 + l] == vertex_id) {
                        std::size_t first = mesh_faces[next * 3 + (l + 1) % 3];
                        adj_verts_via_border[j].push_back(first);
                    }
                }
            }
        }

        is_border[j] = gaps == 1;

        /* Check if vertex is now complex. */
        if (gaps > 1) {
            /* Complex vertex in hole */
            disk_topology = false;
            break;
        }

        if (is_border[j]) {
            idx[j] = num_border_vertices++;
            seed = vertex_id;
        } else {
            idx[j] = j - num_border_vertices;
        }
    }
    tmp.clear();

    /* No disk or genus zero topology */
    if (!disk_topology || num_border_vertices == 0) return false;

    std::vector<std::size_t> border; border.reserve(num_border_vertices);
    std::size_t prev = seed;
    std::size_t curr = seed;
    while (prev == seed || curr != seed) {
        std::size_t next = std::numeric_limits<std::size_t>::max();
        std::vector<std::size_t> const & adj_verts = adj_verts_via_border[g2l[curr]];
        for (std::size_t adj_vert : adj_verts) {
            assert(is_border[g2l[adj_vert]]);
            if (adj_vert != prev && adj_vert != curr) {
                next = adj_vert;
                break;
            }
        }
        if (next != std::numeric_limits<std::size_t>::max()) {
            prev = curr;
            curr = next;
            border.push_back(next);
        } else {
            /* No new border vertex */
            border.clear();
            break;
        }

        /* Loop within border */
        if (border.size() > num_border_vertices) break;
    }

    if (border.size() != num_border_vertices) return false;

    float total_length = 0.0f;
    float total_projection_length = 0.0f;
    for (std::size_t j = 0; j < border.size(); ++j) {
        std::size_t vi0 = border[j];
        std::size_t vi1 = border[(j + 1) % border.size()];
        std::vector<VertexProjectionInfo> vpi0, vpi1;
        #pragma omp critical (vpis)
        {
            vpi0 = vertex_projection_infos->at(vi0);
            vpi1 = vertex_projection_infos->at(vi1);
        }

        /* According to the previous checks (vertex class within the origial
         * mesh and boundary) there has to be at least one projection
         * of each border vertex in a common texture patch. */
        math::Vec2f vp0(NAN), vp1(NAN);
        for (VertexProjectionInfo const & info0 : vpi0) {
            for (VertexProjectionInfo const & info1 : vpi1) {
                if (info0.texture_patch_id == info1.texture_patch_id) {
                    vp0 = info0.projection;
                    vp1 = info1.projection;
                    break;
                }
            }
        }
        assert(!std::isnan(vp0[0]) && !std::isnan(vp0[1]));
        assert(!std::isnan(vp1[0]) && !std::isnan(vp1[1]));

        total_projection_length += (vp0 - vp1).norm();
        math::Vec3f const & v0 = vertices[vi0];
        math::Vec3f const & v1 = vertices[vi1];
        total_length += (v0 - v1).norm();
    }
    float radius = total_projection_length / (2.0f * MATH_PI);

    if (total_length < std::numeric_limits<float>::epsilon()) return false;

    std::vector<math::Vec2f> projections(num_vertices);
    {
        float length = 0.0f;
        for (std::size_t j = 0; j < border.size(); ++j) {
            float angle = 2.0f * MATH_PI * (length / total_length);
            projections[g2l[border[j]]] = math::Vec2f(std::cos(angle), std::sin(angle));
            math::Vec3f const & v0 = vertices[border[j]];
            math::Vec3f const & v1 = vertices[border[(j + 1) % border.size()]];
            length += (v0 - v1).norm();
        }
    }

    typedef Eigen::Triplet<float, int> Triplet;
    std::vector<Triplet> coeff;
    std::size_t matrix_size = num_vertices - border.size();

    Eigen::VectorXf xx(matrix_size), xy(matrix_size);

    if (matrix_size != 0) {
        Eigen::VectorXf bx(matrix_size);
        Eigen::VectorXf by(matrix_size);
        for (std::size_t j = 0; j < num_vertices; ++j) {
            if (is_border[j]) continue;

            std::size_t const vertex_id = l2g[j];

            /* Calculate "Mean Value Coordinates" as proposed by Michael S. Floater */
            std::map<std::size_t, float> weights;

            std::vector<std::size_t> const & adj_faces = mesh_info[vertex_id].faces;
            for (std::size_t adj_face : adj_faces) {
                std::size_t v0 = mesh_faces[adj_face * 3 + 0];
                std::size_t v1 = mesh_faces[adj_face * 3 + 1];
                std::size_t v2 = mesh_faces[adj_face * 3 + 2];
                if (v1 == vertex_id) std::swap(v1, v0);
                if (v2 == vertex_id) std::swap(v2, v0);

                math::Vec3f v01 = vertices[v1] - vertices[v0];
                float v01n = v01.norm();
                math::Vec3f v02 = vertices[v2] - vertices[v0];
                float v02n = v02.norm();

                /* Ensure numerical stability */
                if (v01n * v02n < std::numeric_limits<float>::epsilon()) return false;

                float calpha = v01.dot(v02) / (v01n * v02n);
                float alpha = std::acos(clamp(calpha, -1.0f, 1.0f));
                weights[g2l[v1]] += std::tan(alpha / 2.0f) / (v01n / 2.0f);
                weights[g2l[v2]] += std::tan(alpha / 2.0f) / (v02n / 2.0f);
            }

            std::map<std::size_t, float>::iterator it;
            float sum = 0.0f;
            for (it = weights.begin(); it != weights.end(); ++it)
                sum += it->second;
            if (sum < std::numeric_limits<float>::epsilon()) return false;
            for (it = weights.begin(); it != weights.end(); ++it)
                it->second /= sum;

            bx[idx[j]] = 0.0f;
            by[idx[j]] = 0.0f;
            for (it = weights.begin(); it != weights.end(); ++it) {
                if (is_border[it->first]) {
                    math::Vec2f projection = projections[it->first];
                    bx[idx[j]] += projection[0] * it->second;
                    by[idx[j]] += projection[1] * it->second;
                } else {
                    coeff.push_back(Triplet(idx[j], idx[it->first], -it->second));
                }
            }
        }

        typedef Eigen::SparseMatrix<float> SpMat;
        SpMat A(matrix_size, matrix_size);
        A.setIdentity();

        Eigen::SparseLU<SpMat> solver;
        solver.analyzePattern(A);
        solver.factorize(A);
        xx = solver.solve(bx);
        xy = solver.solve(by);
    }

    float const max_hole_patch_size = MAX_HOLE_PATCH_SIZE;
    int image_size = std::min(std::floor(radius * 1.1f) * 2.0f, max_hole_patch_size);
    /* Ensure a minimum scale of one */
    image_size += 2 * (1 + texture_patch_border);
    int scale = image_size / 2 - texture_patch_border;
    for (std::size_t j = 0, k = 0; j < num_vertices; ++j) {
        if (is_border[j]) {
            projections[j] = projections[j] * scale + image_size / 2;
        } else {
            projections[j] = math::Vec2f(xx[k], xy[k]) * scale + image_size / 2;
            ++k;
        }
    }

    std::vector<math::Vec2f> texcoords; texcoords.reserve(hole.size());
    for (std::size_t const face_id : hole) {
        for (std::size_t j = 0; j < 3; ++j) {
            std::size_t const vertex_id = mesh_faces[face_id * 3 + j];
            math::Vec2f const & projection = projections[g2l[vertex_id]];
            texcoords.push_back(projection);
        }
    }
    mve::FloatImage::Ptr image = mve::FloatImage::create(image_size, image_size, 3);
    //DEBUG image->fill_color(*math::Vec4uc(0, 255, 0, 255));
    TexturePatch::Ptr texture_patch = TexturePatch::create(0, hole, texcoords, image);
    std::size_t texture_patch_id;
    #pragma omp critical
    {
        texture_patches->push_back(texture_patch);
        texture_patch_id = texture_patches->size() - 1;
    }

    for (std::size_t j = 0; j < num_vertices; ++j) {
        std::size_t const vertex_id = l2g[j];
        std::vector<std::size_t> const & adj_faces = mesh_info[vertex_id].faces;
        std::vector<std::size_t> faces; faces.reserve(adj_faces.size());
        for (std::size_t adj_face : adj_faces) {
            if (graph.get_label(adj_face) == 0) {
                faces.push_back(adj_face);
            }
        }
        VertexProjectionInfo info = {texture_patch_id, projections[j], faces};
        #pragma omp critical (vpis)
        vertex_projection_infos->at(vertex_id).push_back(info);
    }

    return true;
}

void
generate_texture_patches(UniGraph const & graph, mve::TriangleMesh::ConstPtr mesh,
    mve::MeshInfo const & mesh_info,
    std::vector<TextureView> * texture_views, Settings const & settings,
    std::vector<std::vector<VertexProjectionInfo> > * vertex_projection_infos,
    std::vector<TexturePatch::Ptr> * texture_patches) {

    util::WallTimer timer;

    mve::TriangleMesh::FaceList const & mesh_faces = mesh->get_faces();
    mve::TriangleMesh::VertexList const & vertices = mesh->get_vertices();
    vertex_projection_infos->resize(vertices.size());

    std::size_t num_patches = 0;

    std::cout << "\tRunning... " << std::flush;
    #pragma omp parallel for schedule(dynamic)
#if !defined(_MSC_VER)
    for (std::size_t i = 0; i < texture_views->size(); ++i) {
#else
    for (std::int64_t i = 0; i < texture_views->size(); ++i) {
#endif

        std::vector<std::vector<std::size_t> > subgraphs;
        int const label = i + 1;
        graph.get_subgraphs(label, &subgraphs);

        TextureView * texture_view = &texture_views->at(i);
        texture_view->load_image();
        std::list<TexturePatchCandidate> candidates;
        for (std::size_t j = 0; j < subgraphs.size(); ++j) {
            candidates.push_back(generate_candidate(label, *texture_view, subgraphs[j], mesh, settings));
        }
        texture_view->release_image();

        /* Merge candidates which contain the same image content. */
        std::list<TexturePatchCandidate>::iterator it, sit;
        for (it = candidates.begin(); it != candidates.end(); ++it) {
            for (sit = candidates.begin(); sit != candidates.end();) {
                Rect<int> bounding_box = sit->bounding_box;
                if (it != sit && bounding_box.is_inside(&it->bounding_box)) {
                    TexturePatch::Faces & faces = it->texture_patch->get_faces();
                    TexturePatch::Faces & ofaces = sit->texture_patch->get_faces();
                    faces.insert(faces.end(), ofaces.begin(), ofaces.end());

                    TexturePatch::Texcoords & texcoords = it->texture_patch->get_texcoords();
                    TexturePatch::Texcoords & otexcoords = sit->texture_patch->get_texcoords();
                    math::Vec2f offset;
                    offset[0] = sit->bounding_box.min_x - it->bounding_box.min_x;
                    offset[1] = sit->bounding_box.min_y - it->bounding_box.min_y;
                    for (std::size_t i = 0; i < otexcoords.size(); ++i) {
                        texcoords.push_back(otexcoords[i] + offset);
                    }

                    sit = candidates.erase(sit);
                } else {
                    ++sit;
                }
            }
        }

        it = candidates.begin();
        for (; it != candidates.end(); ++it) {
            std::size_t texture_patch_id;

            #pragma omp critical
            {
                texture_patches->push_back(it->texture_patch);
                texture_patch_id = num_patches++;
            }

            std::vector<std::size_t> const & faces = it->texture_patch->get_faces();
            std::vector<math::Vec2f> const & texcoords = it->texture_patch->get_texcoords();
            for (std::size_t i = 0; i < faces.size(); ++i) {
                std::size_t const face_id = faces[i];
                std::size_t const face_pos = face_id * 3;
                for (std::size_t j = 0; j < 3; ++j) {
                    std::size_t const vertex_id = mesh_faces[face_pos  + j];
                    math::Vec2f const projection = texcoords[i * 3 + j];

                    VertexProjectionInfo info = {texture_patch_id, projection, {face_id}};

                    #pragma omp critical
                    vertex_projection_infos->at(vertex_id).push_back(info);
                }
            }
        }
    }

    merge_vertex_projection_infos(vertex_projection_infos);

    {
        std::vector<std::size_t> unseen_faces;
        std::vector<std::vector<std::size_t> > subgraphs;
        graph.get_subgraphs(0, &subgraphs);

        #pragma omp parallel for schedule(dynamic)
#if !defined(_MSC_VER)
        for (std::size_t i = 0; i < subgraphs.size(); ++i) {
#else
        for (std::int64_t i = 0; i < subgraphs.size(); ++i) {
#endif
            std::vector<std::size_t> const & subgraph = subgraphs[i];

            bool success = false;
            if (settings.hole_filling) {
                success = fill_hole(subgraph, graph, mesh, mesh_info,
                    vertex_projection_infos, texture_patches);
            }

            if (success) {
                num_patches += 1;
            } else {
                if (settings.keep_unseen_faces) {
                    #pragma omp critical
                    unseen_faces.insert(unseen_faces.end(),
                        subgraph.begin(), subgraph.end());
                }
            }
        }

        if (!unseen_faces.empty()) {
            mve::FloatImage::Ptr image = mve::FloatImage::create(3, 3, 3);
            std::vector<math::Vec2f> texcoords;
            for (std::size_t i = 0; i < unseen_faces.size(); ++i) {
                math::Vec2f projections[] = {{2.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 2.0f}};
                texcoords.insert(texcoords.end(), &projections[0], &projections[3]);
            }
            TexturePatch::Ptr texture_patch
                = TexturePatch::create(0, unseen_faces, texcoords, image);
            texture_patches->push_back(texture_patch);
            std::size_t texture_patch_id = texture_patches->size() - 1;

            for (std::size_t i = 0; i < unseen_faces.size(); ++i) {
                std::size_t const face_id = unseen_faces[i];
                std::size_t const face_pos = face_id * 3;
                for (std::size_t j = 0; j < 3; ++j) {
                    std::size_t const vertex_id = mesh_faces[face_pos  + j];
                    math::Vec2f const projection = texcoords[i * 3 + j];

                    VertexProjectionInfo info = {texture_patch_id, projection, {face_id}};

                    vertex_projection_infos->at(vertex_id).push_back(info);
                }
            }
        }
    }

    merge_vertex_projection_infos(vertex_projection_infos);

    std::cout << "done. (Took " << timer.get_elapsed_sec() << "s)" << std::endl;
    std::cout << "\t" << num_patches << " texture patches." << std::endl;
}

TEX_NAMESPACE_END
