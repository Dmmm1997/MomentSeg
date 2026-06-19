def build_qwen2_5_vl_grounded_conversation_seg_datapipes_with_pixels(json_dir,
                                        image_dir,
                                        batch_size=None,
                                        cycle_count=1e8,
                                        num_to_skip=0,
                                        ind_range=None,
                                        tokenizer=None,
                                        image_transform=None,
                                        processor=None,
                                        min_pixels = 12*12 * 28 * 28,
                                        max_pixels = 30*30 * 28 * 28,
                                        inference_mode=False,
                                        dataset_name=None, 
                                        use_det_prompt=None,
                                        max_length=1e5,
                                        max_mask=1e3,  # 最多出多少个框
                                        use_stat=False,
                                        ignore_crowded=False,    
                                        random_crowded_masks=True,
                                        visualization_prob=0.005, #可视化概率参数
                                        packing=False,
                                        packing_max_seq_lens=10240,
                                        **kwargs,
                                       ):
    """
    datapipe of referseg dataset (such as RefCOCO/+/g/CLEF, COCO instance segmentation, ReasonSeg...)
    """

    def load_from(idx):
        file = os.path.join(json_dir, f"{idx:010d}.json")
        try:
            if not os.path.exists(file):
                file = os.path.join(json_dir, f"annotation_{idx}.json")
                
            with open(file, 'r') as f:
                sample = json.load(f)

            conversations=[]
            # dataset_name = sample["dataset_info"] if dataset_name is None else dataset_name
            roles = ("user", "assistant")

            # TODO: support multiple images
            if isinstance(sample['images'], list):
                img_filename = os.path.basename(sample['images'][0]['file_name'])
                height, width = sample['images'][0]['height'], sample['images'][0]['width']
            else:
                img_filename = os.path.basename(sample['images']['file_name'])
                height, width = sample['images']['height'], sample['images']['width']

            seg_masks = []
            if len(sample['annotations']['grounding_task']) == 1 and sample['annotations']['grounding_task'][0] is None:
                # 如果没有类别的话，就不训练了
                return None
            img_path = os.path.join(image_dir, img_filename.split('/')[-1])

            num_qa = len(sample['annotations']['grounding_task'])
            QA_index = np.random.randint(0, num_qa)
            conv = sample['annotations']['grounding_task'][QA_index]

            # ===================== lvis_muse ======================
            if dataset_name == "lvis_muse":
                question = conv["Q"].strip()
                answer = re.sub(r"\{seg\}", "<SEG>", conv["A"])
                phrases = conv['rephreased_name_list'] if conv['rephreased_name_list'][0] is not None else conv['category_names_list']
                for phrase in phrases:
                    if phrase is not None:
                        answer = re.sub(rf"{phrase}", f"<p> {phrase} </p>", answer)
                matches = re.findall(r"\{seg\}", conv["A"])
                object_num = conv['object_num']
                if object_num != len(matches):
                    # print("the number of {seg} is not matched the number of segmentations")
                    return None
            # ===========================================================================

            # =============== grandf_ha, grandf_flickr, grandf_openpsgcg  ===============
            elif 'grand' in dataset_name.lower():
                question = random.choice(GROUNDED_CONV_QUESTION_PROMPT_LIST)
                pattern = r"\[(.*?)\]\[seg_\d+\]"
                # answer = re.sub(pattern, r"\1 <SEG>", conv["A"])
                answer = re.sub(pattern, r"<p> \1 </p> <SEG>", conv["A"]) 
                matches = re.findall(r"\[seg_\d+\]", conv["A"])
            else:
                raise ValueError(f"dataset_name {dataset_name} is not supported")
            # ===========================================================================

            # add grounded conv seg's task prompt
            suffix_prompt = random.choice(GROUNDED_CONV_SEGMENT_TASK_PROMPT_LIST) 
            if suffix_prompt[0] != ' ':
                question = question + ' ' + suffix_prompt
            else:
                question = question + suffix_prompt
       
            item = {
                "role": roles[0],
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                        "min_pixels": min_pixels, # 该参数可以被process_vision_info函数识别
                        "max_pixels": max_pixels, # 该参数可以被process_vision_info函数识别
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
            conversations.append(item)

            conversations.append({
                "role": roles[1],
                "content": answer})

            all_seg_masks_original = [] # 用于存储原始加载的 mask
            for i, match in enumerate(matches):
                if dataset_name == "lvis_muse":
                    mask_image_path = os.path.join(json_dir.replace('annotation_0', 'grounding_task'), conv[f'seg_{i+1}'].split('/')[-1])
                else:
                    name = match.split(']')[0].split('[')[1]
                    mask_image_path = os.path.join(json_dir.replace('annotation_0', 'grounding_task'), conv[name].split('/')[-1])
                current_mask = np.array(Image.open(mask_image_path).convert('L')) # <-- 加载 mask
                all_seg_masks_original.append(current_mask) # 存储原始 mask
            
            # check the number of masks:
            assistant_responses = " ".join(conv["content"] for conv in conversations if conv["role"] == "assistant")
            seg_count = len(re.findall(r"<SEG>", assistant_responses))
            assert seg_count == len(all_seg_masks_original)

            seg_masks = [(mask > 128).astype(np.float32) for mask in all_seg_masks_original]
            image_input_pil_prcocessed, video_input_  =process_vision_info(conversations)
            sam_images, sam_resizes = prepare_for_sam(image_input_pil_prcocessed, img_size = 1024)

            # 根据image_input_pil_prcocessed的大小，以及缩放率，调整image_input_pil_prcocessed和seg_masks的大小
            # 这里image_input_pil_prcocessed只有一张输入图像 NOTE 这里下采样为1，不resize图像，主要是来resize seg masks
            image_input_pil, seg_masks_resized = resize(image_input_pil_prcocessed[0], seg_masks, downsample=1)

            if not inference_mode and seg_masks is not None:
                seg_masks_resized = [torch.from_numpy(seg_mask) for seg_mask in seg_masks_resized]
                # sam_seg_masks (torch.tensor) [num_instance, resize_h, resize_w]
                sam_seg_masks = torch.stack(seg_masks_resized, dim=0)
            else:
                sam_seg_masks = None
            
             # use the official qwen processor to convert QA; add_generation_prompt=False
            processed_text_question_answer = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
            # import pdb; pdb.set_trace()
            inputs = processor(text=[processed_text_question_answer], 
                               images=[image_input_pil],
                               return_tensors="pt",
                               )

            num_image_tokens = (inputs["image_grid_thw"][0].prod() // 4).item()
            num_tokens = len(inputs["input_ids"][0])

            # this part is just for statitics
            if use_stat:
                processor_name = processor.__class__.__name__
                log_token_stats(dataset_name, processor_name, QA_index, idx, num_tokens, num_image_tokens, min_pixels,
                    max_pixels)

            length = inputs["input_ids"].shape[1]
            # print(f"loaded {file} length: {length} and mask nums: {len(seg_masks)}")
            if inputs["input_ids"].shape[1] > max_length: # drop too long sentence
                num_token = inputs["input_ids"].shape[1]
                print(f'dropping {file} as it has {num_token}, more than {max_length} tokens')
                return None
            if len(seg_masks) > max_mask: # drop samples with too many masks
                print(f'dropping {file} as it has {len(seg_masks)}, more than {max_mask} masks')
                return None
            
            # ==================== 可视化调用点 ====================
            if random.random() < visualization_prob:
                if seg_masks_resized is not None:
                    output_viz_filename = f"{dataset_name}_idx{idx}_masks{len(seg_masks_resized)}.png"
                    # 临时保存 image_input_pil
                    temp_img_path = os.path.join(VISUALIZATION_DIR, f"temp_{idx}.png")
                    image_input_pil.save(temp_img_path)
                    visualize_masks_and_text(
                        temp_img_path,
                        seg_masks_resized, # 使用 resize 后的 mask
                        processed_text_question_answer,
                        output_viz_filename
                    )
                    os.remove(temp_img_path) # 删除临时文件
            # ======================================================
            
            return_dict = {}

            target_ids = inputs["input_ids"].clone()
            # [1, seq_len] 屏蔽非answer
            # import pdb; pdb.set_trace()
            target_ids_supervised_mask = make_target_mask(processor, target_ids).unsqueeze(0)
            target_ids[~target_ids_supervised_mask] = -100

            # 其中image token的数量 = sum(inputs["input_ids"][0]==151655) == (width/14)*(height/14) / 4
            return_dict["input_ids"] = inputs["input_ids"]
            return_dict["target_ids"] = target_ids
            return_dict["pixel_values"] = inputs["pixel_values"] # [(width/14)*(height/14), 1176]
            return_dict["input_ids"] = inputs["input_ids"] 
            return_dict["attention_mask"] = inputs["attention_mask"]
            return_dict["image_grid_thw"] = inputs["image_grid_thw"] # [1, width / 14, height /14]
            return_dict["sam_seg_masks"] = sam_seg_masks
            return_dict["sam_images"] = sam_images
            return_dict["sam_resizes"] = sam_resizes
            return_dict['unique_id'] = dataset_name + '_' + str(idx) + '_' + str(QA_index)  # 使用dataset_name和idx和QAindex组合确保唯一性
            return_dict['seq_length'] = length
            return_dict['dataset_name'] = dataset_name + '-id' + str(idx) + '-l' + str(length) + '-m' + str(len(seg_masks))
            # import pdb; pdb.set_trace()
            return return_dict
    
        except Exception as e:
            # 可能遇到读取图片路径不存在的问题，可以注释掉报错打印
            traceback.print_exc()
            print(f"Error loading file {file}: {e}")
            return None