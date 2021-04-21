import os
import sys
import random
import Model.DBconnect as DBconnect

import justGo as gen

def login(user_name, user_pwd):
    """
    登陆确认，传递进入用户名，用户密码，并将传递进来的数据和数据库中的记录进行比对
    返回出是否成功登陆内容
    Args:
        user_name 用户名
        user_pwd 用户密码

    Returns
        一个字典，返回用户ID，用户名，和用户微信ID
    """
    db = DBconnect.DBconnect()
    info = db.dbQuery_userLogin(user_name, user_pwd)
    if info == None:
        dic = {"returnCode": "r0"}
    else:
        dic = {
            "returnCode": "a0",
            "user_id": info[0],
            "user_name": info[1]
        }
    return dic

def __is_already(db, user_id):
    """
    内部函数，用来判断该用户是否已经存在，该内部方法的调用时刻在于创建【用户ID】的时候进行判断
    （即使user_id生成8位随机数，但还是不排除有可能有重复）

    Args:
        db 数据库打开的指针
        user_id 用户ID

    Return
        False - 不重复， True - 重复
    """
    info = db.dbQuery_user_is_already(user_id)
    if info == None:
        return False
    else:
        return True


def register(user_name, user_pwd, user_email, user_phone):
    """
    创建一个新用户, 通过传递进来的用户名和密码注册。
    自动生成一个8位随机数字的user_id，这个id将会是整个系统中用户的绝对唯一标识符

    Args:
        user_name - 用户登陆ID
        user_pwd - 用户登陆密码
        user_email - 用户邮件
        user_phone - 用户电话号码

    Returns:
        returnCode 正确返回a0，错误返回r0
        user_id 用户ID，通过随机数字生成
        user_name - 用户登陆ID
        user_pwd - 用户登陆密码
        user_email - 用户邮件
        user_phone - 用户电话号码

    """
    db = DBconnect.DBconnect()
    new_user_id = str(random.randint(0, 99999999)).zfill(8)
    bool_is_already = __is_already(db, new_user_id)
    while bool_is_already:
        new_user_id = str(random.randint(0, 99999999)).zfill(8)
        bool_is_already = __is_already(db, new_user_id)

    # 插入数据库
    is_successful = db.dbInsert(
        "user",
        new_user_id,
        user_name,
        user_pwd,
        user_email,
        user_phone
    )
    if is_successful:
        dic = {
            "returnCode": "a0",
            "user_id": new_user_id,
            "user_name": user_name,
            "user_pwd": user_pwd,
            "user_email": user_email,
            "user_phone": user_phone
        }
    else:
        dic = {
            "returnCode": "r0"
        }
    return dic

def modification(user_id, user_name, user_pwd, user_email, user_phone):
    """
    更新用户信息，输入用户ID作为索引
    可以修改的信息有，user_name用户名，user_pwd用户密码，is_admin是管理员么？
    
    user_id 用户ID，通过随机数字生成
    user_name - 用户登陆ID
    user_pwd - 用户登陆密码
    user_email - 用户邮件
    user_phone - 用户电话号码

    返回一个字典，其中包含一个returnCode，当他等于a0的时候表示获取正确信息，返回r0的时候表示获取信息失败
    同时返回的字典还会有基本的查询信息。
    """
    db = DBconnect.DBconnect()

    is_successful = db.dbUpdate_user_infomation(
        user_id, user_name, user_pwd, user_email, user_phone
    )

    if is_successful:
        dic = {
            "returnCode": "a0",
            "user_id": user_id,
            "user_name": user_name,
            "user_pwd": user_pwd,
            "user_email": user_email,
            "user_phone": user_phone
        }
    else:
        dic = {
            "returnCode": "r0"
        }
    return dic




if __name__ == '__main__':
    user_id = "00000001"
    user_name = "xiami"
    user_pwd = "123123"
    user_email = "asdhjkqrw@qq.com"
    user_phone = "18334656778"

    ans = modification(user_id, user_name,user_pwd,user_email,user_phone)
    print(ans)
    #tag = generate_change_tag()
    #user_id = "10000001"
    #truncated = 0.7
    #gen.init_modif(truncated, user_id, tag)