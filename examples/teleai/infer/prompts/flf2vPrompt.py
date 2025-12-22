PROMPT_CONFIGS = {
    "flf2v_02": {
        "short_prompt": "From an aerial perspective, a modern stadium transforms from its initial steel-and-glass structure into a lush 'ecological dome' gradually overtaken by vines, flowers, and trees. The background lighting shifts from daylight to dusk, creating a scene brimming with vitality and a magical ambiance",
        "dense_prompt": "An aerial view captures the transformation of a modern stadium from its initial sleek steel-and-glass structure into a vibrant 'ecological dome' gradually enveloped by lush greenery, vines, and blooming flowers. The background transitions from bright daylight to a warm, golden dusk, enhancing the scene's vitality and magical ambiance. The stadium's roof, initially reflective and angular, now blends seamlessly with the surrounding nature, creating a harmonious blend of urban architecture and natural beauty. The transition from a stark, industrial setting to one rich with life and color is both striking and serene, showcasing a vision of sustainable urban development.",
        "ref_images": {
            "0": "/nvfile-heatstorage/benchmark/video/fl2v_example/FL_1_F_building.png",
            "-1": "/nvfile-heatstorage/benchmark/video/fl2v_example/FL_1_L_building.png"
        }
    },
    "flf2v_00": {
        "short_prompt": "Young woman eating a glazed donut with colored chocolates and a milkshake, on a pink background.",
        "dense_prompt": "A young woman with long, wavy black hair is enjoying a glazed donut adorned with colorful candies and a milkshake against a soft pink backdrop. She wears a striped shirt and a delicate necklace, with a gentle smile as she takes a bite of the donut. The scene transitions to her sipping the milkshake through a straw, her expression one of contentment and satisfaction. The pink background remains consistent throughout, creating a warm and inviting atmosphere. A close-up shot captures her enjoyment of the treats.",
        "ref_images": {
            "0": "/nvfile-heatstorage/benchmark/video/example/mixkit-girl-eating-a-glazed-donut-with-a-milkshake-40831-hd-ready_0.png",
            "-1": "/nvfile-heatstorage/benchmark/video/example/mixkit-girl-eating-a-glazed-donut-with-a-milkshake-40831-hd-ready_249.png"
        }
    },
    "flf2v_01": {
        "short_prompt": "Woman with long brown hair wearing a long white dress walks on the beach towards a group of big black boulders as small waves break on the rocks.",
        "dense_prompt": "A serene and contemplative woman with long, flowing brown hair gracefully walks along the sandy beach towards a cluster of large, dark boulders. She is elegantly dressed in a flowing white gown that billows gently with each step she takes. The soft, muted tones of the sunset paint the sky in hues of pale blue and pink, casting a tranquil glow over the scene. Small waves rhythmically lap against the rocks, creating a soothing soundtrack to her peaceful journey. The camera follows her from behind, capturing the gentle sway of her dress and the tranquil beauty of the ocean and sky.",
        "ref_images": {
            "0": "/nvfile-heatstorage/benchmark/video/example/mixkit-woman-walking-on-beach-towards-boulders-1012-hd-ready_0.png",
            "-1": "/nvfile-heatstorage/benchmark/video/example/mixkit-woman-walking-on-beach-towards-boulders-1012-hd-ready_249.png"
        }
    },
    "FL_motor": {
        "dense_prompt": """The video frames depict a motorcyclist's perspective as they ride along a coastal road. The first - person view shows the motorcycle's dashboard, which is equipped with a digital display showing various gauges and the handlebars with mirrors. The road is a divided one, with a separating wall on the right side and a line marking the lane. A few cars are ahead on the same road, and the sky is clear, indicating a sunny day. As the motorcycle travels, it approaches an intersection where a parking area for small boats is visible, suggesting a pleasant seaside environment. The backdrop features lush green mountains and a turquoise body of water, contributing to a picturesque scene.""",
        "ref_images": {
            "0": "/nvfile-heatstorage/AIGC_H100/ljq/repos/teleai_pipe/test/compare_score/motor_F.jpg",
            "-1": "/nvfile-heatstorage/AIGC_H100/ljq/repos/teleai_pipe/test/compare_score/motor_L.jpg",
        },
    },
    "FL_teapot": {
        "dense_prompt": """The video begins with a silver teapot. Subsequently, a large number of vivid pink flower patterns emerge on its surface. The flowers are interspersed with green branches and leaves, and the color combination is fresh and soft. The background then turns into a blue sky with white clouds, and the plane on which the teapot is placed is light pink. Some small pieces of flowers and plants are scattered around, creating a dreamy and romantic atmosphere.""",
        "ref_images": {
            "0": "/nvfile-heatstorage/AIGC_H100/ljq/repos/distribute_vast/test/assets/FL_7_F_teapot.png",
            "-1": "/nvfile-heatstorage/AIGC_H100/ljq/repos/distribute_vast/test/assets/FL_7_L_teapot.png",
        },
      },
}