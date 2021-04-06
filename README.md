# AnimeFace-StyleGAN2-WebAPI
AnimeFace-DjangoWebAPI（Use StyleGAN2）

本项目为本人的毕业设计

主要参考和应用了NVIDIA-LAB的StyleGAN2算法和https://github.com/halcy/AnimeFaceNotebooks的训练再封装出各种可以调参的tag格式

项目将会作为WebAPI的方式操作，通过FLASK-API传输参数和图片（不方便放在服务器中部署，因为需要消耗GPU资源才能生成图片）

### 整体项目分层：

![Architecture](.\doc\pic\Architecture.png)

1. Model层 - 与数据库的交互，包含增删改查等基本功能
2. StyleGAN层 - 算法层，包含算法的生成，文件等等内容
3. Controller层 - 控制层，控制Model层与StyleGAN层的传输效果，归宗信息传递给API层
4. API层 - 与Web端交互





### 参数包含（不保证具体顺序是这个）

hair_between_eyes
smile
long_hair        
upper_body       
sketch
short_hair       
simple_background
closed_mouth     
eyebrows
brown_hair       
shirt
bow
open_mouth       
eyelashes
ribbon
white_background
lips
red_eyes
brown_eyes
hair_bow
shiny_hair
parted_lips
collarbone
white_shirt
blonde_hair
sidelocks
hair_ornament
circle_cut
shiny
blue_eyes
animal_ears
purple_eyes
:d
:o
hat
yellow_eyes
jewelry
hair_ribbon
orange_eyes
bare_shoulders
traditional_media
rain
twintails
tears
ahoge
black_hair
green_eyes
signature
heart
purple_hair
fang
pink_hair
holding
shikishi
pink_eyes
nose_blush
collared_shirt
red_hair
gloves
eyes_visible_through_hair
orange_hair
pink_lips
expressionless
silver_hair
long_sleeves
outdoors
blunt_bangs
dress
choker
teeth
frills
striped
pink_background
wing_collar
cat_ears
school_uniform
grey_background
gradient_background
sleeveless
symbol-shaped_pupils
hair_intakes
blue_background
hairband
red_bow
breasts
photo
transparent_background
fingernails
aqua_eyes
food
gradient
ponytail
necktie
light_brown_hair
flower
yellow_background
jacket

