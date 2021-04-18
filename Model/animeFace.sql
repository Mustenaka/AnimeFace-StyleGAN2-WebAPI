/*
 Navicat Premium Data Transfer

 Source Server         : localMySQL
 Source Server Type    : MySQL
 Source Server Version : 80023
 Source Host           : localhost:3306
 Source Schema         : animeFace

 Target Server Type    : MySQL
 Target Server Version : 80023
 File Encoding         : 65001

 Date: 08/04/2021 16:24:20
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for savePath
-- ----------------------------
DROP TABLE IF EXISTS `savePath`;
CREATE TABLE `savePath` (
  `saveid` bigint(8) unsigned zerofill NOT NULL AUTO_INCREMENT COMMENT '引用级联外键 - 用户信息表',
  `randomPath` varchar(255) COLLATE utf8_bin DEFAULT NULL COMMENT '随机生成路径',
  `realFacePath` varchar(255) COLLATE utf8_bin DEFAULT NULL COMMENT '真实照片路径',
  `generatePath` varchar(255) COLLATE utf8_bin DEFAULT NULL COMMENT '生成路径',
  PRIMARY KEY (`saveid`),
  CONSTRAINT `saveid` FOREIGN KEY (`saveid`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8 COLLATE=utf8_bin;

-- ----------------------------
-- Table structure for user
-- ----------------------------
DROP TABLE IF EXISTS `user`;
CREATE TABLE `user` (
  `id` bigint(8) unsigned zerofill NOT NULL AUTO_INCREMENT COMMENT '用户ID',
  `username` varchar(255) COLLATE utf8_bin DEFAULT NULL COMMENT '用户名',
  `password` varchar(255) COLLATE utf8_bin DEFAULT NULL COMMENT '密码',
  `email` varchar(255) COLLATE utf8_bin DEFAULT NULL COMMENT '邮件',
  `phone` varchar(255) COLLATE utf8_bin DEFAULT NULL COMMENT '电话号码',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8 COLLATE=utf8_bin;

SET FOREIGN_KEY_CHECKS = 1;
